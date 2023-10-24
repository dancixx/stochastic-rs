use std::time::Instant;

use indicatif::ProgressBar;
use stochastic_rs::{
    prelude::*,
    processes::fbm::{fbm, Fbm},
};

fn main() {
    // let start = Instant::now();
    // let pb = ProgressBar::new(10000);
    // for _ in 0..10000 {
    //     let _ = fbm(0.7, 2500, None, None);
    //     pb.inc(1);
    // }
    // pb.finish();
    // println!("Time elapsed: {:?}", start.elapsed());

    // let start = Instant::now();
    // let _ = par_fbm(10000, 0.7, 2500, None, None);
    // println!("Time elapsed: {:?}", start.elapsed());

    let start = Instant::now();
    let _ = Fbm::new(0.7, 5000, None, Some(100000), None).sample_par();
    println!("Time elapsed: {:?}", start.elapsed());

    // CIR
    // let mut plot = Plot::new();
    // let path = || cir::cir(1.0, 0.6, 0.5, 1000, None, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // FCIR
    // let mut plot = Plot::new();
    // let path = || cir::cir(1.0, 1.2, 0.2, 1000, None, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // OU
    // let mut plot = Plot::new();
    // let path = || ou::ou(1.0, 1.2, 0.2, 1000, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // FOU
    // let mut plot = Plot::new();
    // let path = || ou::fou(0.7, 1.0, 1.2, 0.2, 1000, None, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // let mut plot = Plot::new();
    // let path = || ou::ou(1.0, 1.2, 0.2, 1000, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // FOU
    // let mut plot = Plot::new();
    // let path = || ou::fou(0.7, 1.0, 1.2, 0.2, 1000, None, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // GBM
    // let mut plot = Plot::new();
    // let path = || gbm::gbm(0.1, 0.3, 200, None, None);

    // plot.add_trace(Scatter::new((0..200).collect::<Vec<usize>>(), path()));
    // plot.show();

    // FGBM
    // let mut plot = Plot::new();
    // let path = || gbm::fgbm(0.2, 0.1, 0.3, 1000, None, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // Jacobi
    // let mut plot = Plot::new();
    // let path = || jacobi::jacobi(1.0, 1.2, 0.2, 1000, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // FJacobi
    // let mut plot = Plot::new();
    // let path = || ou::fou(0.7, 1.0, 1.2, 0.2, 1000, None, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // Variance Gamma
}
