//
// Gamma Test, based on ()  
//
use num_traits::Float;

pub struct GammaTest<'a, T : Float> {
    pub intercept : Option<T>,
    pub slope : Option<T>,
    pub v_ratio : Option<T>,
    pub p : usize,
    pub output_variance : Option<T>,
    //pub mean_absolute_error : Option<f32>,
    pub sigmas : Option<Vec<T>>,
    pub deltas : Option<Vec<T>>,
    pub inputs : &'a [Vec<T>],
    pub output : &'a [T],
    pub euclidean_distance_table : Vec<Vec<T>>,
    pub y_distance_table : Vec<Vec<T>>,
    pub near_neighbor_table : Vec<Vec<T>>,
    mm : usize,
    mmt : T,
    m: usize,
}


impl<'a, T:Float> GammaTest<'a, T>{

    pub fn new(inputs : &'a [Vec<T>], output : &'a [T], p : usize)->GammaTest<'a, T> {
            let n = output.len();
            let mm = inputs.len();
            if n == 0 {panic!("output.len() must greater than 0!");}
            if mm == 0 {panic!("inputs.len() must greater than 0!");}
            if p == 0 {panic!("p must greater than 0!");}
            
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
                output,
                euclidean_distance_table,
                y_distance_table,
                near_neighbor_table,    
                mm,
                mmt,
                m,
            }
        }
    
        pub fn compute(&mut self){
            self.initialize_eulidean_distance_table();
            
            let (ydt, nnt) = self.sort_neighbours();
            
            //println!("yDistanceTable : \n {:?}", ydt);
    
            //println!("nearNeighbourTable : \n {:?}", nnt);
    
            self.compute_sigmas(&nnt.as_slice());
    
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
                    ydistancetable[i][k] = T::abs(self.output[near_neighbour_index]-self.output[i]); 
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
    
        fn compute_deltas(&mut self, ydistancetable:  &Vec<Vec<T>>) {
          
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
                    let output_variance = GammaTest::compute_variance(&self.output);
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
    
       /*  fn compute_mean_absolute_error(&mut self){
            let mut reg_deltas : Vec<f32> = Vec::new();
            
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
        } */
       
    }
    
 
    

    




pub struct GammaTestf32 {
    ///
    /// intercept (Gamma) value
    /// 
    pub intercept : Option<f32>,
    pub slope : Option<f32>,
    pub v_ratio : Option<f32>,
    pub p : usize,
    pub output_variance : Option<f32>,
    //pub mean_absolute_error : Option<f32>,
    pub sigmas : Option<Vec<f32>>,
    pub deltas : Option<Vec<f32>>,
    pub inputs : Vec<Vec<f32>>,
    pub output : Vec<f32>,
    pub euclidean_distance_table : Vec<Vec<f32>>,
    pub y_distance_table : Vec<Vec<f32>>,
    pub near_neighbor_table : Vec<Vec<f32>>,
    mm : usize,
    m: usize,
}

impl GammaTestf32{
    pub fn new(inputs : Vec<Vec<f32>>, output : Vec<f32>, p : usize)->Self {
        let n = output.len();
        let mm = inputs.len();
        if n == 0 {panic!("output.len() must greater than 0!");}
        if mm == 0 {panic!("inputs.len() must greater than 0!");}
        if p == 0 {panic!("p must greater than 0!");}
        
        if mm!=n {panic!("inputs & output lengths must be equal!");}
        
        let m = inputs[0].len();
      
        let euclidean_distance_table : Vec<Vec<f32>> = Vec::with_capacity(mm);
        let y_distance_table : Vec<Vec<f32>> = Vec::with_capacity(mm);
        let near_neighbor_table : Vec<Vec<f32>> = Vec::with_capacity(mm);
        GammaTestf32 {
            intercept : None,
            slope : None,
            v_ratio : None,
            p,
            output_variance : None,
            //mean_absolute_error : None,
            sigmas : None,
            deltas : None,
            inputs,
            output,
            euclidean_distance_table,
            y_distance_table,
            near_neighbor_table,    
            mm,
            m
        }
    }

    pub fn compute(&mut self){
        self.initialize_eulidean_distance_table();
        
        let (ydt, nnt) = self.sort_neighbours();
        
        //println!("yDistanceTable : \n {:?}", ydt);

        //println!("nearNeighbourTable : \n {:?}", nnt);

        self.compute_sigmas(&nnt.as_slice());

        //println!("Sigmas : \n {:?}", self.sigmas);

        self.compute_deltas(&ydt.as_slice());

        //println!("deltas : \n {:?}", self.deltas);

        self.do_regression(); 
        
        //self.compute_mean_absolute_error();

        self.compute_v_ratio();

    } 

    fn initialize_eulidean_distance_table(&mut self){

        for _i in 0..self.mm{
            let mut row : Vec<f32> = Vec::with_capacity(self.mm);
            for _j in 0..self.mm{
                row.push(0.0);
            }
            self.euclidean_distance_table.push(row);        
        }

        //println!("ec_table = {:?}", self.euclidean_distance_table.len());
            
        for i in 0..self.mm{
                
            for j in (i+1)..self.mm{
                
                let mut dist : f32 =0.0f32;

                for k in 0..self.m {
                    dist += (self.inputs[i][k]-self.inputs[j][k]).powi(2); 
                }
                self.euclidean_distance_table[i][j]=f32::sqrt(dist);
            }
        }

        for j in 0..self.mm{
            for i in (j+1)..self.mm {
                self.euclidean_distance_table[i][j]= self.euclidean_distance_table[j][i]
            }    
        }

        println!("ec_table = {:?}", self.euclidean_distance_table.len());
        
    }

    fn sort_neighbours(&self)->(Vec<Vec<f32>>, Vec<Vec<f32>>) {

        let mut lowerbound : Vec<f32> = Vec::with_capacity(self.mm);
        for _i in 0..self.mm{
            lowerbound.push(0.0f32);
        }    

        let mut ydistancetable: Vec<Vec<f32>> = Vec::with_capacity(self.mm);
        for _i in 0..self.mm {
            let mut row : Vec<f32> = Vec::with_capacity(self.p);
            for _j in 0..self.p {
                row.push(0.0f32);
            }
            ydistancetable.push(row);
        }

        let mut near_neighbour_table: Vec<Vec<f32>> = Vec::with_capacity(self.mm);
        for _i in 0..self.mm {
            let mut row : Vec<f32> = Vec::with_capacity(self.p);
            for _j in 0..self.p {
                row.push(0.0f32);
            }
            near_neighbour_table.push(row);
        }

        let mut currentlewest : f32 = f32::MAX;
        let mut near_neighbour_index : usize =0;
        
        for k in 0..self.p {
        for i in 0..self.mm {
                for j in 0..self.mm {
                    if j!=i {
                        let distance = self.euclidean_distance_table[i][j];
                        if distance >lowerbound[i] && distance < currentlewest {
                            currentlewest = distance;
                            near_neighbour_index = j;
                        }
                    }
                }
                ydistancetable[i][k] = f32::abs(self.output[near_neighbour_index]-self.output[i]); 
                near_neighbour_table[i][k] = currentlewest;
                lowerbound[i] = currentlewest;
                currentlewest = f32::MAX;
        }
        }

        (ydistancetable, near_neighbour_table)

    }

    fn compute_sigmas(&mut self, near_neighbour_table : &[Vec<f32>]) {

        let mut sigmas : Vec<f32> = Vec::with_capacity(self.p);

        for k in 0..self.p {
            let mut sum : f32 = 0.0;
            for i in 0..self.mm{
                sum += f32::powi(near_neighbour_table[i][k],2);
            }
            let sigma = sum/self.mm as f32;
            sigmas.push(sigma);
        }
        self.sigmas = Some(sigmas);
    }

    fn compute_deltas(&mut self, ydistancetable:  &[Vec<f32>]) {
      
        let mut deltas : Vec<f32> = Vec::with_capacity(self.p);
        let n : f32 = 2.0*self.mm as f32;

        for k in 0..self.p{
            
            let mut sum : f32 =0.0;
            for i in 0..self.mm {
                sum += f32::powi(ydistancetable[i][k], 2);
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
                let output_variance = GammaTestf32::compute_variance(&self.output);
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

    fn linear_regression(&self, x : &[f32], y : &[f32])->(f32, f32) {
       
        let nx= x.len();
        let ny = y.len();
        
        if nx != ny {panic!("no equal items!");}

        let sumx : f32 = x.iter().fold(0.0, |acc :f32, v| acc + v);
        let sumy : f32 = y.iter().fold(0.0, |acc, v| acc + v);
        let avgx: f32 = sumx/nx as f32;
        let avgy: f32 = sumy/ny as f32;

        let mut difx : Vec<f32>  =Vec::with_capacity(nx);
        let mut difx2 : Vec<f32>  =Vec::with_capacity(nx);
        let mut dify : Vec<f32>  =Vec::with_capacity(nx);
        let mut product : Vec<f32>  =Vec::with_capacity(nx);

        for i in 0..nx {
            difx.push(avgx - x[i]);
            difx2.push(f32::powi(difx[i], 2));
            dify.push(avgy - y[i]);
            product.push(difx[i]*dify[i]);
        }

        let sumdifx2 :f32 = difx2.iter().fold(0.0, |acc, v| acc+v);
        let sumxy : f32 = product.iter().fold(0.0, |acc, v| acc+v);

         let slope =sumxy/sumdifx2; 
         let intercept = avgy - (slope*avgx);

        (slope, intercept)
    }
    
    pub fn compute_variance(data : &[f32])->Option<f32>{
        let n = data.len();
        match n {
             0 => None,
            _ => {
                let sum = data.iter().fold(0.0, |acc, x| acc+x);
                let avg = sum / n as f32;

                let difference = data.iter().fold(0.0, |acc, x| acc + f32::powi(x-avg, 2));
                let variance = difference /n as f32;
                Some(variance)
            }   
        }
  
    }

   /*  fn compute_mean_absolute_error(&mut self){
        let mut reg_deltas : Vec<f32> = Vec::new();
        
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
    } */
   
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
    
        //println!("m = {}, mm = {}", gt.m, gt.mm);
        
        gt.compute();
    
        //println!("euclidean_distance_table : \n {:?}", gt.euclidean_distance_table);
    
        //println!("slope = {:?}", gt.slope);
    
        //println!("intercept = {:?}", gt.intercept);

        assert_eq!(gt.slope,  Some(33.54095));
        assert_eq!(gt.intercept, Some(20.578278));

    } 



    #[test]
    fn gammatestf32_compute_test_1(){
        let output = [54.0f32, 30.0, 3.0, 28.0];
        let inputs =[
            [3.0f32, 4.0, 4.0].to_vec(),
            [2.0f32, 1.0, 3.0].to_vec(),
            [1.0f32, 0.0, 1.0].to_vec(),
            [1.0f32, 1.0, 1.0].to_vec(),
            ];
    
        let p : usize = 3;
    
        let mut gt = GammaTestf32::new(inputs.to_vec(), output.to_vec(), p);
    
        //println!("m = {}, mm = {}", gt.m, gt.mm);
        
        gt.compute();
    
        //println!("euclidean_distance_table : \n {:?}", gt.euclidean_distance_table);
    
        //println!("slope = {:?}", gt.slope);
    
        //println!("intercept = {:?}", gt.intercept);

        assert_eq!(gt.slope,  Some(33.54095));
        assert_eq!(gt.intercept, Some(20.578278));

    } 




    #[test]
    fn test_variance(){
        let data = [46.0f32, 69.0, 32.0,60.0, 52.0, 41.0];
        let variance = GammaTestf32::compute_variance(&data);
        assert_eq!(variance, Some(147.666667));
    }

}
