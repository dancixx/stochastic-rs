use std::{borrow::BorrowMut, cell::RefCell, f64::consts::PI};

use candle_core::{DType, Device, IndexOp, Result, Shape, Tensor};
use candle_nn::{
  layer_norm, linear, linear_no_bias, ops::dropout, seq, Activation, Dropout, LayerNorm,
  LayerNormConfig, Linear, Module, Sequential, VarBuilder,
};
use candle_transformers::models::mimi::transformer::StreamingMultiheadAttention;

pub struct Time2Vec {
  seq_len: usize,
  embed_dim: usize,
  wb: Tensor,
  bb: Tensor,
  wa: Tensor,
  ba: Tensor,
}

impl Time2Vec {
  pub fn new(seq_len: usize, embed_dim: usize, device: &Device) -> Result<Self> {
    let wb = Tensor::zeros((embed_dim,), DType::F32, device)?;
    let bb = Tensor::zeros((embed_dim,), DType::F32, device)?;
    let wa = Tensor::zeros((embed_dim,), DType::F32, device)?;
    let ba = Tensor::zeros((embed_dim,), DType::F32, device)?;

    Ok(Self {
      seq_len,
      embed_dim,
      wb,
      bb,
      wa,
      ba,
    })
  }
}

impl Module for Time2Vec {
  fn forward(&self, xs: &Tensor) -> Result<Tensor> {
    // x shape: (batch_size, seq_len, input_dim)
    let batch_size = xs.shape().dims()[0];
    let tt = Tensor::arange(0f32, self.seq_len as f32, xs.device())?.unsqueeze(1)?;
    let tt = tt.repeat(&[1, self.embed_dim])?;
    let v = (&self.wb * &tt + &self.bb + (&self.wa * &tt + &self.ba)?.sin()?)?;
    let v = v
      .unsqueeze(0)?
      .expand(&[batch_size, self.seq_len, self.embed_dim])?;
    Ok(v)
  }
}

pub struct FeedForward {
  pub net: Sequential,
  pub dropout_rate: f32,
}

impl FeedForward {
  pub fn new(n_embd: usize, dropout_rate: f32, vs: VarBuilder) -> Result<Self> {
    let linear1 = linear(n_embd, 4 * n_embd, vs.pp("feedforward_linear1"))?;
    let new_gelu = Activation::NewGelu;
    let linear2 = linear(4 * n_embd, n_embd, vs.pp("feedforward_linear2"))?;
    let dropout = Dropout::new(dropout_rate);

    let net = seq()
      .add(linear1)
      .add(new_gelu)
      .add(linear2)
      .add_fn(move |xs| Ok(dropout.forward(&xs, true).unwrap()));

    Ok(Self { net, dropout_rate })
  }
}

impl Module for FeedForward {
  fn forward(&self, xs: &Tensor) -> Result<Tensor> {
    let xs = self.net.forward(xs)?;
    Ok(xs)
  }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L62
#[derive(Debug, Clone)]
struct MultiHeadAttention {
  query: Linear,
  key: Linear,
  value: Linear,
  out: Linear,
  n_head: usize,
  span: tracing::Span,
  softmax_span: tracing::Span,
  matmul_span: tracing::Span,
  kv_cache: Option<(Tensor, Tensor)>,
}

impl MultiHeadAttention {
  fn new(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
    let span = tracing::span!(tracing::Level::TRACE, "multi-head-attn");
    let softmax_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-softmax");
    let matmul_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-matmul");
    let query = linear(n_state, n_state, vb.pp("q_proj"))?;
    let value = linear(n_state, n_state, vb.pp("v_proj"))?;
    let key = linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
    let out = linear(n_state, n_state, vb.pp("out_proj"))?;
    Ok(Self {
      query,
      key,
      value,
      out,
      n_head,
      span,
      softmax_span,
      matmul_span,
      kv_cache: None,
    })
  }

  fn forward(
    &mut self,
    x: &Tensor,
    xa: Option<&Tensor>,
    mask: Option<&Tensor>,
    flush_cache: bool,
  ) -> Result<Tensor> {
    let _enter = self.span.enter();
    let q = self.query.forward(x)?;
    let (k, v) = match xa {
      None => {
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;
        (k, v)
      }
      Some(x) => {
        if flush_cache {
          self.kv_cache = None;
        }
        if let Some((k, v)) = &self.kv_cache {
          (k.clone(), v.clone())
        } else {
          let k = self.key.forward(x)?;
          let v = self.value.forward(x)?;
          self.kv_cache = Some((k.clone(), v.clone()));
          (k, v)
        }
      }
    };
    let wv = self.qkv_attention(&q, &k, &v, mask)?;
    let out = self.out.forward(&wv)?;
    Ok(out)
  }

  fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
    let (n_batch, n_ctx, n_state) = x.dims3()?;
    let target_dims = &[n_batch, n_ctx, self.n_head, n_state / self.n_head];
    x.reshape(target_dims)?.transpose(1, 2)
  }

  fn qkv_attention(
    &self,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
  ) -> Result<Tensor> {
    let (_, n_ctx, n_state) = q.dims3()?;
    let scale = ((n_state / self.n_head) as f64).powf(-0.25);
    let q = (self.reshape_head(q)? * scale)?;
    let k = (self.reshape_head(k)?.transpose(2, 3)? * scale)?;
    let v = self.reshape_head(v)?.contiguous()?;
    let mut qk = {
      let _enter = self.matmul_span.enter();
      q.matmul(&k)?
    };
    if let Some(mask) = mask {
      let mask = mask.i((0..n_ctx, 0..n_ctx))?;
      qk = qk.broadcast_add(&mask)?
    }
    let w = {
      let _enter = self.softmax_span.enter();
      candle_nn::ops::softmax_last_dim(&qk)?
    };
    let wv = {
      let _enter = self.matmul_span.enter();
      w.matmul(&v)?
    }
    .transpose(1, 2)?
    .flatten_from(2)?;
    Ok(wv)
  }

  fn reset_kv_cache(&mut self) {
    self.kv_cache = None;
  }
}

pub struct Block {
  sa: RefCell<MultiHeadAttention>,
  ffwd: FeedForward,
  ln1: LayerNorm,
  ln2: LayerNorm,
}

impl Block {
  pub fn new(n_embd: usize, n_head: usize, dropout_rate: f32, vs: VarBuilder) -> Result<Self> {
    let sa = MultiHeadAttention::new(n_embd, n_head, vs.pp("sa"))?;
    let ffwd = FeedForward::new(n_embd, dropout_rate, vs.pp("ffwd"))?;
    let ln1 = layer_norm(n_embd, LayerNormConfig::default(), vs.pp("ln1"))?;
    let ln2 = layer_norm(n_embd, LayerNormConfig::default(), vs.pp("ln2"))?;
    Ok(Self {
      sa: RefCell::new(sa),
      ffwd,
      ln1,
      ln2,
    })
  }
}

impl Module for Block {
  fn forward(&self, xs: &Tensor) -> Result<Tensor> {
    let x_norm = self.ln1.forward(xs)?;
    let attn_output = self
      .sa
      .borrow_mut()
      .forward(&x_norm, Some(&x_norm), Some(&x_norm), false)?;
    let xs = (xs + attn_output)?;
    let xs = (&xs + self.ffwd.forward(&self.ln2.forward(&xs)?)?)?;
    Ok(xs)
  }
}

pub struct TransformerEncoder {
  input_linear: candle_nn::Linear,
  time2vec: Time2Vec,
  blocks: Vec<Block>,
  ln: candle_nn::LayerNorm,
  fc_mu: candle_nn::Linear,
  fc_log_var: candle_nn::Linear,
  fc_volatility: candle_nn::Linear,
}

impl TransformerEncoder {
  pub fn new(
    input_dim: usize,
    n_embd: usize,
    n_head: usize,
    n_layers: usize,
    latent_dim: usize,
    seq_len: usize,
    dropout_rate: f32,
    vs: VarBuilder,
  ) -> Result<Self> {
    let input_linear = linear(input_dim, n_embd, vs.pp("input_linear"))?;
    let time2vec = Time2Vec::new(seq_len, n_embd, &Device::Cpu)?;
    let mut blocks = Vec::new();
    for i in 0..n_layers {
      blocks.push(Block::new(
        n_embd,
        n_head,
        dropout_rate,
        vs.pp(&format!("blocks_{}", i)),
      )?);
    }
    let ln = layer_norm(
      n_embd,
      LayerNormConfig::default(),
      vs.pp("transformer_encoder_layer_norm"),
    )?;
    let fc_mu = linear(n_embd, latent_dim, vs.pp("transformer_encoder_fc_mu"))?;
    let fc_log_var = linear(n_embd, latent_dim, vs.pp("transformer_encoder_fc_log_var"))?;
    let fc_volatility = linear(n_embd, 1, vs.pp("transformer_encoder_fc_volatility"))?;
    Ok(Self {
      input_linear,
      time2vec,
      blocks,
      ln,
      fc_mu,
      fc_log_var,
      fc_volatility,
    })
  }

  pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    // xs shape: (batch_size, seq_len, input_dim)
    let xs_input = self.input_linear.forward(xs)?;
    let t2v = self.time2vec.forward(xs)?;
    let mut xs = (xs_input + t2v)?;
    xs = xs.transpose(0, 1)?; // Shape: (seq_len, batch_size, n_embd)
    for block in &self.blocks {
      xs = block.forward(&xs)?;
    }
    xs = self.ln.forward(&xs)?;
    xs = xs.transpose(0, 1)?; // Shape: (batch_size, seq_len, n_embd)
    let xs_pooled = xs.mean(1)?; // Average pooling over the sequence length
    let mu = self.fc_mu.forward(&xs_pooled)?;
    let log_var = self.fc_log_var.forward(&xs_pooled)?;
    let sigma_estimated = self.fc_volatility.forward(&xs_pooled)?;

    Ok((mu, log_var, sigma_estimated))
  }
}

pub struct TransformerDecoder {
  latent_linear: candle_nn::Linear,
  blocks: Vec<Block>,
  ln: candle_nn::LayerNorm,
  output_linear: candle_nn::Linear,
  seq_len: usize,
}

impl TransformerDecoder {
  pub fn new(
    latent_dim: usize,
    n_embd: usize,
    n_head: usize,
    n_layers: usize,
    output_dim: usize,
    seq_len: usize,
    dropout_rate: f32,
    vs: VarBuilder,
  ) -> Result<Self> {
    let latent_linear = linear(
      latent_dim,
      n_embd,
      vs.pp("transfomer_decoder_latent_linear"),
    )?;
    let mut blocks = Vec::new();
    for i in 0..n_layers {
      blocks.push(Block::new(
        n_embd,
        n_head,
        dropout_rate,
        vs.pp(&format!("blocks_{}", i)),
      )?);
    }
    let ln = layer_norm(
      n_embd,
      LayerNormConfig::default(),
      vs.pp("transfomer_decoder_layer_norm"),
    )?;
    let output_linear = linear(
      n_embd,
      output_dim,
      vs.pp("transfomer_decoder_output_linear"),
    )?;
    Ok(Self {
      latent_linear,
      blocks,
      ln,
      output_linear,
      seq_len,
    })
  }

  pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
    // z shape: (batch_size, latent_dim)
    let batch_size = z.shape().dims()[0];
    let z_projected = self.latent_linear.forward(z)?; // Shape: (batch_size, n_embd)
    let z_expanded = z_projected.unsqueeze(1)?.expand(&[
      batch_size,
      self.seq_len,
      z_projected.shape().dims()[1],
    ])?;
    let mut x = z_expanded.transpose(0, 1)?; // Shape: (seq_len, batch_size, n_embd)
    for block in &self.blocks {
      x = block.forward(&x)?;
    }
    x = self.ln.forward(&x)?;
    x = x.transpose(0, 1)?; // Shape: (batch_size, seq_len, n_embd)
    let x_reconstructed = self.output_linear.forward(&x)?;
    Ok(x_reconstructed)
  }
}

pub struct TransformerVAE {
  encoder: TransformerEncoder,
  decoder: TransformerDecoder,
}

impl TransformerVAE {
  pub fn new(
    input_dim: usize,
    n_embd: usize,
    n_head: usize,
    n_layers_enc: usize,
    n_layers_dec: usize,
    latent_dim: usize,
    seq_len: usize,
    dropout_rate: f32,
    vs: VarBuilder,
  ) -> Result<Self> {
    let encoder = TransformerEncoder::new(
      input_dim,
      n_embd,
      n_head,
      n_layers_enc,
      latent_dim,
      seq_len,
      dropout_rate,
      vs.pp("encoder"),
    )?;
    let decoder = TransformerDecoder::new(
      latent_dim,
      n_embd,
      n_head,
      n_layers_dec,
      input_dim,
      seq_len,
      dropout_rate,
      vs.pp("decoder"),
    )?;
    Ok(Self { encoder, decoder })
  }

  fn reparameterize(&self, mu: &Tensor, log_var: &Tensor) -> Result<Tensor> {
    let std = (log_var * 0.5)?.exp()?;
    let eps = Tensor::randn(0.0, 1.0, std.shape(), std.device())?;
    Ok((mu + &(&eps * &std)?)?)
  }

  pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let (mu, log_var, sigma_estimated) = self.encoder.forward(xs)?;
    let z = self.reparameterize(&mu, &log_var)?;
    let x_reconstructed = self.decoder.forward(&z)?;
    Ok((x_reconstructed, sigma_estimated, mu, log_var, z))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use candle_core::{DType, Device, Result, Tensor};
  use candle_nn::VarMap;

  #[test]
  fn test_transformer_vae_forward() -> Result<()> {
    let seq_len = 10;
    let input_dim = 5;
    let n_embd = 32;
    let n_head = 4;
    let n_layers_enc = 2;
    let n_layers_dec = 2;
    let latent_dim = 16;
    let dropout_rate = 0.1;

    let device = &Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F64, device);

    let model = TransformerVAE::new(
      input_dim,
      n_embd,
      n_head,
      n_layers_enc,
      n_layers_dec,
      latent_dim,
      seq_len,
      dropout_rate,
      vs,
    )?;

    let batch_size = 32;
    let xs = Tensor::randn(0.0, 1.0, &[batch_size, seq_len, input_dim], device)?;

    let (x_reconstructed, sigma_estimated, mu, log_var, z) = model.forward(&xs)?;

    assert_eq!(
      x_reconstructed.shape().dims(),
      &[batch_size, seq_len, input_dim]
    );
    assert_eq!(sigma_estimated.shape().dims(), &[batch_size, 1]);
    assert_eq!(mu.shape().dims(), &[batch_size, latent_dim]);
    assert_eq!(log_var.shape().dims(), &[batch_size, latent_dim]);
    assert_eq!(z.shape().dims(), &[batch_size, latent_dim]);

    Ok(())
  }
}
