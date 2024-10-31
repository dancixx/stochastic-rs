use candle_core::{DType, Device, Result, Tensor, D};
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
    let logits = model.forward(&x_train)?;
    let loss = loss::mse(&logits, &y_train)?;
    adam.backward_step(&loss)?;

    let test_logits = model.forward(&x_test)?;
    let test_loss = loss::mse(&test_logits, &y_test)?;

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
  use ndarray::{array, s, Array1, Array2, Axis};
  use ndarray_npy::read_npy;
  use tempfile::NamedTempFile;

  use crate::ai::{utils::train_test_split_for_array2, volatility::heston::train, DataSet};

  #[test]
  fn fit_surface() -> Result<()> {
    let file = File::open("src/ai/volatility/HestonTrainSet.txt.gz").unwrap();
    let mut decoder = GzDecoder::new(file);

    let mut temp_file = NamedTempFile::new().unwrap();
    std::io::copy(&mut decoder, &mut temp_file).unwrap();

    let data: Array2<f64> = read_npy(temp_file.path()).unwrap();

    let xx = data.slice(s![.., 0..5]).to_owned();
    let yy = data.slice(s![.., 5..]).to_owned();

    let strikes = array![0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5];
    let maturities = array![0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0];

    let (x_train, x_test, y_train, y_test) = train_test_split_for_array2(xx, yy, 0.15, None);

    #[derive(Debug)]
    struct StandardScaler {
      mean: Array1<f64>,
      std: Array1<f64>,
    }

    impl StandardScaler {
      fn fit(data: &Array2<f64>) -> Self {
        let mean = data.mean_axis(Axis(0)).unwrap();
        let std = data.std_axis(Axis(0), 0.0);
        StandardScaler { mean, std }
      }

      fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        data - &self.mean + &self.std
      }

      fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        data * &self.std + &self.mean
      }
    }

    let scale = StandardScaler::fit(&y_train);
    let scale2 = StandardScaler::fit(&x_train);

    let y_train_transform = scale.transform(&y_train);
    let y_test_transform = scale.transform(&y_test);
    let x_train_transform = scale2.transform(&x_train);
    let x_test_transform = scale2.transform(&x_test);

    let _x_inv_transform = scale2.inverse_transform(&x_test_transform);

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

    let y_train_scaled = y_train.map_axis(Axis(1), |row| myscale(&row.to_owned(), &lb, &ub));
    let y_test_scaled = y_test.map_axis(Axis(1), |row| myscale(&row.to_owned(), &lb, &ub));

    let x_train_f32 = x_train_transform.mapv(|v| v as f32);
    let y_train_f32 = y_train_transform.mapv(|v| v as f32);
    let x_test_f32 = x_test_transform.mapv(|v| v as f32);
    let y_test_f32 = y_test_transform.mapv(|v| v as f32);

    let x_train = Tensor::from_slice(x_train_f32.as_slice().unwrap(), (10200, 5), &Device::Cpu)?;
    let y_train = Tensor::from_slice(y_train_f32.as_slice().unwrap(), (10200, 88), &Device::Cpu)?;
    let x_test = Tensor::from_slice(x_test_f32.as_slice().unwrap(), (1800, 5), &Device::Cpu)?;
    let y_test = Tensor::from_slice(y_test_f32.as_slice().unwrap(), (1800, 88), &Device::Cpu)?;

    let model = train(
      DataSet {
        x_train,
        y_train,
        x_test,
        y_test,
      },
      &Device::Cpu,
      5,
      30,
      88,
      200,
    )
    .unwrap();

    Ok(())
  }
}
