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
    let linear2 = linear(hidden_size, 10, vs.pp("linear-2"))?;
    let linear3 = linear(hidden_size, 10, vs.pp("linear-3"))?;
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
    let xs = self.linear1.forward(&xs)?;
    let xs = xs.elu(2.0)?;
    let xs = self.linear2.forward(&xs)?;
    let xs = xs.elu(2.0)?;
    let xs = self.linear3.forward(&xs)?;
    let xs = xs.elu(2.0)?;
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
  let mut final_acc = 0.0;

  for epoch in 1..epoch + 1 {
    let logits = model.forward(&x_train)?;
    let loss = loss::mse(&logits, &y_train)?;
    adam.backward_step(&loss)?;

    let test_logits = model.forward(&x_test)?;
    let sum_ok = test_logits
      .argmax(D::Minus1)?
      .eq(&y_test)?
      .to_dtype(DType::F32)?
      .sum_all()?
      .to_scalar::<f32>()?;
    let test_acc = sum_ok / y_test.dims1()? as f32;
    final_acc = 100.0 * test_acc;
    println!(
      "Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
      loss.to_scalar::<f32>()?,
      final_acc
    );
    if final_acc == 100.0 {
      break;
    }
  }

  if final_acc < 100.0 {
    Err(candle_core::Error::Msg(String::from(
      "The model is not trained well enough.",
    )))
  } else {
    Ok(model)
  }
}
