use stochastic_rs::{noises::gn, processes::bm::bm};

fn main() {
    let noise = bm(1000, 1);

    println!("{:?}", noise);
    // println!("{:?}", gn.len());
}
