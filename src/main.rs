use stochastic_rs::processes::{bm, fbm};

fn main() {
    let fbm = fbm::fbm_cholesky(10, 0.7, Some(2));
    println!("fbm: {:?}", fbm[0]);

    let bm = bm::bm(10, None);
    println!("bm: {:?}", bm[0]);
}
