# gammatest

## Definition 
[gammatest](https://github.com/SaadDAHMANI/gammatest) is a simple [rust](https://github.com/rust-lang/rust) implementation of the Gamma Test. 

Gamma Test [1] is non-parametric test for feature selection frequently used in machine learning.

The [gammatest](https://github.com/SaadDAHMANI/gammatest) crate is based on [the paper](https://ijssst.info/Vol-06/No-1&2/Kemp.pdf) [2].

### References 
[1] Stefánsson, A., Končar, N., & Jones, A. J. (1997). A note on the gamma test. Neural Computing & Applications, 5(3), 131-133.

[2] Kemp, S. E., Wilson, I. D., & Ware, J. A. (2004). A tutorial on the gamma test. International Journal of Simulation: Systems, Science and Technology, 6(1-2), 67-75.

## Example

```rust
use gammatest::*;
   
fn main()
   {    
      // Suppose we would perform Gamma Test on the model y=f(x1, x2, x3)
      // Give the input matrix (x1, x2, x3), where x1, x2, and x3 are grouped as rows. 
      let inputs =[//x1,    x2,  x3 
                  [3.0f32, 4.0, 4.0].to_vec(),
                  [2.0f32, 1.0, 3.0].to_vec(),
                  [1.0f32, 0.0, 1.0].to_vec(),
                  [1.0f32, 1.0, 1.0].to_vec(),
               ];

      // Give the output vector (y values) 
      let output = [54.0f32, 30.0, 3.0, 28.0];
      
      // p is the number of neighbors 
      let p : usize = 3;
      
      // Build the GammaTest using f32 data type
      let mut gt : GammaTest<f32> = GammaTest::new(&inputs, &output, p);
      
      // To use f64 data type 
      //let mut gt : GammaTest<f64> = GammaTest::new(&inputs, &output, p);

      // Call function compute() to compute GammaTest parameters.
      gt.compute();

      // Check results
      assert_eq!(gt.slope,  Some(33.54095));
      assert_eq!(gt.intercept, Some(20.578278));
    } 
 ``` 




## Current development state

In the current version, [gammatest](https://github.com/SaadDAHMANI/gammatest) uses the "Brute force approach" to sort k-near neighbors, which is a simple but slow method comparing to some others.

