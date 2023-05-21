use stochastic_rs::noises::gn;

fn main() {
    let noise = gn::gn(1000, 1);

    println!("{:?}", noise);
    // println!("{:?}", gn.len());
}
