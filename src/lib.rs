//!
//! [gammatest](https://github.com/SaadDAHMANI/gammatest) is a simple implementation of the Gamma Test.
//! Gamma Test \[1\] is non-parametric test for feature selection frequently used in machine learning.
//! The [gammatest](https://github.com/SaadDAHMANI/gammatest) crate is based on [the paper](https://ijssst.info/Vol-06/No-1&2/Kemp.pdf) \[2\].
//! 
//! ## References 
//! \[1\] Stefánsson, A., Končar, N., & Jones, A. J. (1997). A note on the gamma test. Neural Computing & Applications, 5(3), 131-133.
//! \[2\] Kemp, S. E., Wilson, I. D., & Ware, J. A. (2004). A tutorial on the gamma test. International Journal of Simulation: Systems, Science and Technology, 6(1-2), 67-75.

use num_traits::Float;

///
/// GammaTest struct computes Gamma Test for a given (inputs/outputs) dataset in lazy way.
///
/// # Example :
/// ```
/// use gammatest::*;
/// fn main()
///    {    
///       // Give the input matrix
///       let inputs =[
///                   [3.0f32, 4.0, 4.0].to_vec(),
///                   [2.0f32, 1.0, 3.0].to_vec(),
///                   [1.0f32, 0.0, 1.0].to_vec(),
///                   [1.0f32, 1.0, 1.0].to_vec(),
///                ];
/// 
///       // Give the output vector 
///       let output = [54.0f32, 30.0, 3.0, 28.0];
///
///       // p is the number of neighbors 
///       let p : usize = 3;
///      
///       // Build the GammaTest using f32 data type
///       let mut gt : GammaTest<f32> = GammaTest::new(&inputs, &output, p);
///   
///       // To use f64 data type 
///       //let mut gt : GammaTest<f64> = GammaTest::new(&inputs, &output, p);
///
///       // Call function compute() to compute GammaTest parameters.
///       gt.compute();
///
///       // Check results
///       assert_eq!(gt.slope,  Some(33.54095));
///       assert_eq!(gt.intercept, Some(20.578278));
///     } 
///  ``` 

pub struct GammaTest<'a, T : Float> {
    
    /// The Gamma parameter (intercept)   
    pub intercept : Option<T>,

    /// The slope value
    pub slope : Option<T>,

    /// The V_ratio value 
    pub v_ratio : Option<T>,
 
    /// The number of neighbors
    pub p : usize,

    pub output_variance : Option<T>,
    //pub mean_absolute_error : Option<f32>,
    pub sigmas : Option<Vec<T>>,
    pub deltas : Option<Vec<T>>,
    pub inputs : &'a [Vec<T>],
    pub outputs : &'a [T],
    pub euclidean_distance_table : Vec<Vec<T>>,
    pub y_distance_table : Vec<Vec<T>>,
    pub near_neighbor_table : Vec<Vec<T>>,
    mm : usize,
    mmt : T,
    m: usize,
}

impl<'a, T:Float> GammaTest<'a, T> {
    ///
    /// Build a new Gammatest object. "p" in number of neighbors.
    /// 
    pub fn new(inputs : &'a [Vec<T>], outputs : &'a [T], p : usize)->GammaTest<'a, T> {
            let n = outputs.len();
            let mm = inputs.len();
            if n == 0 {panic!("output.len() must greater than 0!");}
            if mm == 0 {panic!("inputs.len() must greater than 0!");}
                        
            if p < 2 {panic!("the parametre p must be : p >= 2!");}
            if p > mm-1 {panic!("the parametre p must less than inputs (and output) lenght !");}
                        
            if mm!=n {panic!("inputs & output lengths must be equal!");}
            
            let m = inputs[0].len();
          
             let mmt : T = match T::from(mm) {
                None => T::zero(),
                Some(x)=> x,
             };

            let euclidean_distance_table : Vec<Vec<T>> = Vec::with_capacity(mm);
            let y_distance_table : Vec<Vec<T>> = Vec::with_capacity(mm);
            let near_neighbor_table : Vec<Vec<T>> = Vec::with_capacity(mm);
            
            GammaTest {
                intercept : None,
                slope : None,
                v_ratio : None,
                p,
                output_variance : None,
                //mean_absolute_error : None,
                sigmas : None,
                deltas : None,
                inputs,
                outputs,
                euclidean_distance_table,
                y_distance_table,
                near_neighbor_table,    
                mm,
                mmt,
                m,
            }
        }
   
    ///
    /// Call this function to compute Gamma Test values (Slope, Intercep and V_ratio).
    ///  
    pub fn compute(&mut self){
            
            self.initialize_eulidean_distance_table();
            
            let (ydt, nnt) = self.sort_neighbours();
            
            //println!("yDistanceTable : \n {:?}", ydt);
    
            //println!("nearNeighbourTable : \n {:?}", nnt);
    
            self.compute_sigmas(&nnt);
    
            //println!("Sigmas : \n {:?}", self.sigmas);
    
            self.compute_deltas(&ydt);
    
            //println!("deltas : \n {:?}", self.deltas);
    
            self.do_regression(); 
            
            //self.compute_mean_absolute_error();
    
            self.compute_v_ratio();
    
        } 
    
        fn initialize_eulidean_distance_table(&mut self){
    
            for _i in 0..self.mm{
                let mut row : Vec<T> = Vec::with_capacity(self.mm);
                for _j in 0..self.mm{
                    row.push(T::zero());
                }
                self.euclidean_distance_table.push(row);        
            }
    
            //println!("ec_table = {:?}", self.euclidean_distance_table.len());
                
            for i in 0..self.mm{
                    
                for j in (i+1)..self.mm{
                    
                    let mut dist : T = T::zero();
    
                    for k in 0..self.m {
                        dist = dist + (self.inputs[i][k]-self.inputs[j][k]).powi(2); 
                    }
                    self.euclidean_distance_table[i][j]=T::sqrt(dist);
                }
            }
    
            for j in 0..self.mm{
                for i in (j+1)..self.mm {
                    self.euclidean_distance_table[i][j]= self.euclidean_distance_table[j][i]
                }    
            }
    
            //println!("Euclidian_table = {:?}", self.euclidean_distance_table.len());
            
        }
    
        fn sort_neighbours(&self)->(Vec<Vec<T>>, Vec<Vec<T>>) {
    
            let mut lowerbound : Vec<T> = Vec::with_capacity(self.mm);
            for _i in 0..self.mm{
                lowerbound.push(T::zero());
            }    
    
            let mut ydistancetable: Vec<Vec<T>> = Vec::with_capacity(self.mm);
            for _i in 0..self.mm {
                let mut row : Vec<T> = Vec::with_capacity(self.p);
                for _j in 0..self.p {
                    row.push(T::zero());
                }
                ydistancetable.push(row);
            }
    
            let mut near_neighbour_table: Vec<Vec<T>> = Vec::with_capacity(self.mm);
            for _i in 0..self.mm {
                let mut row : Vec<T> = Vec::with_capacity(self.p);
                for _j in 0..self.p {
                    row.push(T::zero());
                }
                near_neighbour_table.push(row);
            }
    
            let mut currentlewest : T = T::max_value();
            let mut near_neighbour_index : usize =0;
            
            for k in 0..self.p {
            for i in 0..self.mm {
                    for j in 0..self.mm {
                        if j!=i {
                            let distance = self.euclidean_distance_table[i][j];
                            if distance > lowerbound[i] && distance < currentlewest {
                                currentlewest = distance;
                                near_neighbour_index = j;
                            }
                        }
                    }
                    ydistancetable[i][k] = T::abs(self.outputs[near_neighbour_index]-self.outputs[i]); 
                    near_neighbour_table[i][k] = currentlewest;
                    lowerbound[i] = currentlewest;
                    currentlewest = T::max_value();
            }
            }
    
            (ydistancetable, near_neighbour_table)
    
        }
    
        fn compute_sigmas(&mut self, near_neighbour_table : &[Vec<T>]) {
    
            let mut sigmas : Vec<T> = Vec::with_capacity(self.p);
            
            for k in 0..self.p {
                let mut sum : T = T::zero();
                for i in 0..self.mm{
                    sum = sum + T::powi(near_neighbour_table[i][k],2);
                }
                let sigma = sum/self.mmt;

                sigmas.push(sigma);
            }
            self.sigmas = Some(sigmas);
        }
    
        fn compute_deltas(&mut self, ydistancetable:  &[Vec<T>]) {
          
            let mut deltas : Vec<T> = Vec::with_capacity(self.p);
            let n : T = T::from(2.0).unwrap()*self.mmt;
    
            for k in 0..self.p{
                
                let mut sum : T = T::zero();

                for i in 0..self.mm {
                    sum = sum + T::powi(ydistancetable[i][k], 2);
                }
                let delta = sum /n;
                deltas.push(delta); 
            }  
            self.deltas = Some(deltas);
        }
    
        fn do_regression(&mut self) {
            
            match &self.sigmas{
                None => {
                    self.slope = None; 
                    self.intercept = None;
                 },
    
                 Some(sigmas)=> {
    
                    match &self.deltas{
                        None => {self.slope = None; self.intercept = None;},
                        Some(deltas) => {
                           let (slope, intercept) = self.linear_regression(sigmas, deltas);                 
                            self.slope = Some(slope);
                            self.intercept = Some(intercept);
                        }
                    }
                 }
            }     
        }
    
        fn compute_v_ratio(&mut self){
            match self.intercept{
                None => self.v_ratio = None,
                Some(gamma) => {
                    let output_variance = GammaTest::compute_variance(self.outputs);
                    self.output_variance = output_variance;
                    match self.output_variance {
                        None => self.v_ratio = None,
                        Some(variance)=> {
                            self.v_ratio = Some(gamma/variance);    
                        }
                    }
                    
                }
            }
        }
    
        fn linear_regression(&self, x : &[T], y : &[T])->(T, T) {
           
            let nx = x.len();
            let ny = y.len();

            let nxt= match T::from(x.len()){
                None => T::one(),
                Some(nxt) => nxt,
            };

            let nyt: T = match T::from(y.len()) {
                None => T::one(),
                Some(nyt) => nyt,
            };
            
            if nx != ny {panic!("no equal items!");}
    
            let sumx : T = x.iter().fold(T::zero(), |acc, v| *v + acc);
            let sumy : T = y.iter().fold(T::zero(), |acc, v| *v + acc);
            let avgx: T = sumx/nxt;
            let avgy: T = sumy/nyt;
    
            let mut difx : Vec<T>  =Vec::with_capacity(nx);
            let mut difx2 : Vec<T>  =Vec::with_capacity(nx);
            let mut dify : Vec<T>  =Vec::with_capacity(nx);
            let mut product : Vec<T>  =Vec::with_capacity(nx);
    
            for i in 0..nx {
                difx.push(avgx - x[i]);
                difx2.push(T::powi(difx[i], 2));
                dify.push(avgy - y[i]);
                product.push(difx[i]*dify[i]);
            }
    
            let sumdifx2 :T = difx2.iter().fold(T::zero(), |acc, v| *v + acc);
            let sumxy : T = product.iter().fold(T::zero(), |acc, v| *v + acc);
    
             let slope =sumxy/sumdifx2; 
             let intercept = avgy - (slope*avgx);
    
            (slope, intercept)
        }
        
        pub fn compute_variance(data : &[T])->Option<T>{
            let n = data.len();
            let nt : T = match T::from(n) {
                None => T::one(),
                Some(nt) => nt,
            };

            match n {
                 0 => None,
                _ => {
                    let sum = data.iter().fold(T::zero(), |acc, x| *x + acc);
                    let avg = sum / nt;
                    let difference :T = data.iter().fold(T::zero(), |acc, x| acc + T::powi(*x-avg, 2));
                    let variance = difference /nt;
                    Some(variance)
                }   
            }
      
        }
    
         /*fn compute_mean_absolute_error(&mut self){
            let mut reg_deltas : Vec<T> = Vec::new();
            
            match &self.sigmas{
                None => self.mean_absolute_error = None, 
                Some(sigmas)=>{ 
    
                    for sigm in sigmas.iter(){
                        let delt = self.slope.unwrap()*sigm + self.intercept.unwrap();
                        reg_deltas.push(delt);
                        
                        let sum = reg_deltas.iter().fold(0.0, |acc, y| acc+y);
                        let avg = sum/reg_deltas.len() as f32;
                        let sumerr = reg_deltas.iter().fold(0.0, |acc, y| acc+ f32::abs(y-avg));
                        self.mean_absolute_error = Some(sumerr/reg_deltas.len() as f32);                    
                    }
                }
            }        
        }*/
       
    }
    
 
mod stat{
    use num_traits::Float;

    #[allow(dead_code)]
    pub enum DataTyp{
        Sample,
        Population,    
    }    
    
    #[allow(dead_code)]
    pub fn standard_deviation<T : Float>(data : &[T], data_type : DataTyp)->T{
        let sum : T =  data.iter().fold(T::zero(), |acc, x| acc + *x);
        let m = data.len() as f32;
               
        let n :T = match T::from(m) {
            None => T::one(),
            Some(n) => n       
        };
    
        let mean = sum/n;
    
        let sumdiff = data.iter().fold(T::zero(), |acc, x| acc + T::powi(*x-mean, 2));
        
        match data_type {
            DataTyp::Sample => T::sqrt(sumdiff/(n-T::one())),
            DataTyp::Population => T::sqrt(sumdiff/n)
        }    
    }
    
    ///
    /// Compute the Standard Error (SE) of a given vector (Sample / Population data).
    /// 
    /// 
    #[allow(dead_code)]
    pub fn standard_error<T : Float>(data : &[T], data_type : DataTyp)->T{
           
        let n = match T::from(data.len()){
            None => T::one(),
            Some(n)=> n
        };
    
        let sdev = standard_deviation(&data, data_type);
        sdev/T::sqrt(n)            
    }     
    
}    

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn gammatest_compute_test_1(){
        let output = [54.0f32, 30.0, 3.0, 28.0];
        let inputs =[
            [3.0f32, 4.0, 4.0].to_vec(),
            [2.0f32, 1.0, 3.0].to_vec(),
            [1.0f32, 0.0, 1.0].to_vec(),
            [1.0f32, 1.0, 1.0].to_vec(),
            ];
    
        let p : usize = 3;
    
        let mut gt : GammaTest<f32> = GammaTest::new(&inputs, &output, p);
           
        gt.compute();
    
        assert_eq!(gt.slope,  Some(33.54095));
        assert_eq!(gt.intercept, Some(20.578278));

    } 


/*     #[test]
    fn gammatest_compute_test_2(){
        let output = [54.0f32, 30.0, 3.0, 28.0, 12.5, 3.5, 5.3, 19.27];
        let inputs =[
            [3.0f32, 4.0, 4.0, 3.2, 3.0].to_vec(), //1
            [2.0f32, 1.0, 3.0, 3.2, 3.0].to_vec(), //2
            [1.0f32, 0.0, 1.0, 3.2, 3.0].to_vec(), //3 
            [1.0f32, 1.0, 1.0, 3.2, 3.0].to_vec(), //4
            [1.0f32, 1.2, 1.1, 3.2, 3.0].to_vec(), //5
            [1.0f32, 1.0, 1.5, 3.2, 3.5].to_vec(), //6
            [1.0f32, 1.7, 1.0, 3.2, 3.5].to_vec(), //7
            [1.2f32, 11.7, 11.0, 13.2, 4.5].to_vec(), //8

            ];
    
        let p : usize = 4;
    
        let mut gt : GammaTest<f32> = GammaTest::new(&inputs, &output, p);
    
        //println!("m = {}, mm = {}", gt.m, gt.mm);
        
        gt.compute();
    
        //println!("euclidean_distance_table : \n {:?}", gt.euclidean_distance_table);
    
        //println!("slope = {:?}", gt.slope);
    
        //println!("intercept = {:?}", gt.intercept);

        assert_eq!(gt.slope,  Some(33.54095));
        assert_eq!(gt.intercept, Some(20.578278));

    } 

 */
    
    #[test]
    fn gammatest_compute_test_3_good_model(){

        let outputs = [0.9f32, 0.9, 0.9, 0.9];
        let inputs = [
            [0.1f32, 0.4, 0.4].to_vec(),
            [0.2f32, 0.35, 0.35].to_vec(),
            [0.3f32, 0.3, 0.3].to_vec(),
            [0.4f32, 0.25, 0.25].to_vec(),
        ];

        let p : usize = 3;
        let mut gt : GammaTest<f32> = GammaTest::new(&inputs, & outputs, p);

        gt.compute();

        assert_eq!(gt.intercept, Some(0.0f32));
        assert_eq!(gt.slope, Some(0.0f32));
    
    }


}
