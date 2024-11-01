use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, loss, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use crate::ai::DataSet;

/// Calibration model for Heston model
///
/// https://github.com/amuguruza/NN-StochVol-Calibrations/blob/master/Heston/NNHeston.ipynb
///
pub struct Model {
  linear1: Linear,
  linear2: Linear,
  linear3: Linear,
  output_layer: Linear,
  epoch: usize,
}

impl Model {
  #[must_use = "new is necessary to create a new instance of Model"]
  pub fn new(
    vs: VarBuilder,
    input_dim: usize,
    hidden_size: usize,
    output_dim: usize,
    epoch: usize,
  ) -> Result<Self> {
    let linear1 = linear(input_dim, hidden_size, vs.pp("linear-1"))?;
    let linear2 = linear(hidden_size, hidden_size, vs.pp("linear-2"))?;
    let linear3 = linear(hidden_size, hidden_size, vs.pp("linear-3"))?;
    let output_layer = linear(hidden_size, output_dim, vs.pp("linear-4"))?;

    Ok(Self {
      linear1,
      linear2,
      linear3,
      output_layer,
      epoch,
    })
  }
}

impl Module for Model {
  fn forward(&self, xs: &Tensor) -> Result<Tensor> {
    let xs = self.linear1.forward(&xs)?.elu(2.0)?;
    let xs = self.linear2.forward(&xs)?.elu(2.0)?;
    let xs = self.linear3.forward(&xs)?.elu(2.0)?;
    let xs = self.output_layer.forward(&xs)?;
    Ok(xs)
  }
}

pub fn train(
  dataset: DataSet,
  device: &Device,
  input_dim: usize,
  hidden_size: usize,
  output_dim: usize,
  epoch: usize,
) -> Result<Model> {
  let x_train = dataset.x_train.to_device(device)?;
  let y_train = dataset.y_train.to_device(device)?;
  let varmap = VarMap::new();
  let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
  let model = Model::new(vs, input_dim, hidden_size, output_dim, epoch)?;
  let mut adam = AdamW::new(varmap.all_vars(), ParamsAdamW::default())?;
  let x_test = dataset.x_test.to_device(device)?;
  let y_test = dataset.y_test.to_device(device)?;

  for epoch in 1..epoch + 1 {
    let logits = model.forward(&y_train)?;
    let loss = loss::mse(&logits, &x_train)?;
    adam.backward_step(&loss)?;

    let test_logits = model.forward(&y_test)?;
    let test_loss = loss::mse(&test_logits, &x_test)?;

    println!(
      "Epoch: {epoch:3} Train loss: {:8.5} Test loss: {:8.5}",
      loss.to_scalar::<f32>()?,
      test_loss.to_scalar::<f32>()?
    );
  }

  Ok(model)
}

#[cfg(test)]
mod tests {
  use std::fs::File;

  use candle_core::{Device, Result, Tensor};
  use candle_nn::Module;
  use flate2::read::GzDecoder;
  use ndarray::{array, s, stack, Array1, Array2, Axis};
  use ndarray_npy::read_npy;
  use plotly::{
    common::{Mode, Title},
    layout::{GridPattern, LayoutGrid},
    Layout, Plot, Scatter,
  };
  use tempfile::NamedTempFile;

  use crate::ai::{utils::train_test_split_for_array2, volatility::heston::train, DataSet};

  #[test]
  fn fit_surface() -> Result<()> {
    let file = File::open("src/ai/volatility/HestonTrainSet.txt.gz").unwrap();
    let mut decoder = GzDecoder::new(file);

    let mut temp_file = NamedTempFile::new().unwrap();
    std::io::copy(&mut decoder, &mut temp_file).unwrap();

    let data: Array2<f64> = read_npy(temp_file.path()).unwrap();

    let xx = data.slice(s![.., 0..5]).to_owned(); // (12000, 5)
    let yy = data.slice(s![.., 5..]).to_owned(); // (12000, 88)

    let strikes = array![0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5];
    let maturities = array![0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0];

    // Shape: (10200, 5), (1800, 5), (10200, 88), (1800, 88)
    let split_data = train_test_split_for_array2(&[yy, xx], 0.15, None);
    let (x_train, x_test) = &split_data[0];
    let (y_train, y_test) = &split_data[1];

    #[derive(Debug)]
    struct StandardScaler2D {
      mean: Array1<f64>,
      std: Array1<f64>,
    }

    impl StandardScaler2D {
      fn fit(data: &Array2<f64>) -> Self {
        let mean = data.mean_axis(Axis(0)).unwrap();
        let std = data.std_axis(Axis(0), 0.0);
        Self { mean, std }
      }

      fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        data - &self.mean + &self.std
      }

      fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        data * &self.std + &self.mean
      }
    }

    let _scale = StandardScaler2D::fit(&y_train);
    let scale2 = StandardScaler2D::fit(&x_train);

    let x_transform = |x_train: &Array2<f64>, x_test: &Array2<f64>| {
      (scale2.transform(x_train), scale2.transform(x_test))
    };
    let (x_train_transform, x_test_transform) = x_transform(x_train, x_test); // (10200, 88), (1800, 88)
    let _x_inverse_transform = |x: &Array2<f64>| scale2.inverse_transform(x);

    let ub = [0.04, -0.1, 1.0, 0.2, 10.0];
    let lb = [0.0001, -0.95, 0.01, 0.01, 1.0];

    fn myscale(x: &Array1<f64>, lb: &[f64; 5], ub: &[f64; 5]) -> Array1<f64> {
      Array1::from(
        (0..5)
          .map(|i| (x[i] - (ub[i] + lb[i]) * 0.5) * 2.0 / (ub[i] - lb[i]))
          .collect::<Vec<f64>>(),
      )
    }

    fn myinverse(x: &Array1<f64>, lb: &[f64; 5], ub: &[f64; 5]) -> Array1<f64> {
      Array1::from(
        (0..5)
          .map(|i| x[i] * (ub[i] - lb[i]) * 0.5 + (ub[i] + lb[i]) * 0.5)
          .collect::<Vec<f64>>(),
      )
    }

    fn scale_array(array: &Array2<f64>, lb: &[f64; 5], ub: &[f64; 5]) -> Array2<f64> {
      let scaled_rows: Vec<Array1<f64>> = array
        .axis_iter(Axis(0))
        .map(|row| myscale(&row.to_owned(), lb, ub))
        .collect();

      stack(
        Axis(0),
        &scaled_rows.iter().map(|row| row.view()).collect::<Vec<_>>(),
      )
      .expect("Failed to stack arrays")
    }

    let y_train_transform = scale_array(&y_train, &lb, &ub);
    let y_test_transform = scale_array(&y_test, &lb, &ub);

    let model = train(
      DataSet {
        x_train: Tensor::from_slice(
          x_train_transform.mapv(|v| v as f32).as_slice().unwrap(),
          (10200, 88),
          &Device::Cpu,
        )?,
        y_train: Tensor::from_slice(
          y_train_transform.mapv(|v| v as f32).as_slice().unwrap(),
          (10200, 5),
          &Device::Cpu,
        )?,
        x_test: Tensor::from_slice(
          x_test_transform.mapv(|v| v as f32).as_slice().unwrap(),
          (1800, 88),
          &Device::Cpu,
        )?,
        y_test: Tensor::from_slice(
          y_test_transform.mapv(|v| v as f32).as_slice().unwrap(),
          (1800, 5),
          &Device::Cpu,
        )?,
      },
      &Device::Cpu,
      5,
      30,
      88,
      5,
    )
    .unwrap();

    fn plot_results_with_plotly(
      strikes: &Array1<f64>,
      maturities: &Array1<f64>,
      x_sample: &Array1<f64>,
      prediction: &Array1<f64>,
    ) -> Result<()> {
      let strikes_dim = strikes.len();
      let maturities_dim = maturities.len();
      let s0 = 1.0;

      let mut plot = Plot::new();

      for i in 0..maturities_dim {
        let start = i * strikes_dim;
        let end = start + strikes_dim;

        // Log-moneyness kiszámítása
        let log_moneyness: Vec<f64> = strikes.iter().map(|&k| (k / s0).ln()).collect();

        // Szeletek az adatokhoz
        let x_sample_slice = x_sample.slice(s![start..end]).to_vec();
        let prediction_slice = prediction.slice(s![start..end]).to_vec();

        // Kék vonal az input adatokhoz
        let trace_input = Scatter::new(log_moneyness.clone(), x_sample_slice)
          .name(format!("Input Data - Maturity {:.2}", maturities[i]))
          .mode(Mode::Lines)
          .line(plotly::common::Line::new().color("blue"))
          .x_axis(format!("x{}", i + 1))
          .y_axis(format!("y{}", i + 1));

        // Piros szaggatott vonal az előrejelzésekhez
        let trace_prediction = Scatter::new(log_moneyness.clone(), prediction_slice)
          .name(format!("NN Approx - Maturity {:.2}", maturities[i]))
          .mode(Mode::Lines)
          .line(
            plotly::common::Line::new()
              .dash(plotly::common::DashType::Dash)
              .color("red"),
          )
          .x_axis(format!("x{}", i + 1))
          .y_axis(format!("y{}", i + 1));

        plot.add_trace(trace_input);
        plot.add_trace(trace_prediction);
      }

      // Layout beállítása gridhez
      let layout = Layout::new()
        .grid(
          LayoutGrid::new()
            .rows(2) // Módosíthatod a kívánt grid elrendezésre
            .columns((maturities_dim + 1) / 2)
            .pattern(GridPattern::Independent),
        )
        .title(Title::from("Implied Volatility Surface"));

      plot.set_layout(layout);
      plot.show();

      Ok(())
    }

    let sample_idx: usize = 1250;
    let x_sample = x_test.slice(s![sample_idx, ..]).to_owned(); // Array1<f64>
    let y_sample = y_test_transform.slice(s![sample_idx, ..]).to_owned(); // Array1<f64>
    let y_sample_tensor = Tensor::from_slice(
      y_sample.mapv(|v| v as f32).as_slice().unwrap(),
      (1, y_sample.len()),
      &Device::Cpu,
    )?;
    let y_pred_tensor = model.forward(&y_sample_tensor)?;
    let y_pred_vec = y_pred_tensor.get(0)?.to_vec1::<f32>()?;
    let prediction = Array1::from(y_pred_vec).mapv(|v| v as f64);

    plot_results_with_plotly(&strikes, &maturities, &x_sample, &prediction)?;

    Ok(())
  }
}
