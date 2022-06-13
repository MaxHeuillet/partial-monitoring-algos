import geometry_v3
import numpy as np


class BPM:

    def __init__(self, game, horizon, p0, sigma0 ):
      self.game = game
      self.horizon = horizon

      self.N = game.n_actions
      self.M = game.n_outcomes
      self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)
      print('n-actions', self.N, 'n-outcomes', self.M, 'alphabet', self.A)

      self.SignalMatrices = [ np.array( [ [1,1] ] ), np.array( [ [0,1], [1,0] ] ) ] #geometry_v3.calculate_signal_matrices(game.FeedbackMatrix, self.N, self.M, self.A)
      self.sstInvs = [  np.linalg.inv( s @ s.T ) for s in  self.SignalMatrices ] 
      self.Ps = [ self.SignalMatrices[i].T @  self.sstInvs[i] @ self.SignalMatrices[i] for i in range(self.N) ] 

      self.p0 = p0 
      self.sigma0 = sigma0
      self.p = p0 
      self.sigma = sigma0
      self.sigmaInv = np.identity(2)
      self.sample_num = 100
      self.feedback_counter = np.zeros( ( self.N, self.A ) )

      self.sample_size = 1000
      self.numC = 10
      self.samples = np.zeros( ( self.sample_size * self.numC * 2, self.M) )
      self.cInequality = self.getCells()
      self.ActiveActions = None
      self.n = np.zeros(self.N)

    def populateSamples(self, t):
      for i in range(self.sample_size):
        x = np.random.uniform(0,1, self.M)

        stocX = x / x.sum()
        xnormMat = ( (stocX - self.p ).T @ self.sigmaInv @ ( stocX - self.p) ) / np.log(t)
        samp1 = np.zeros(self.M)
        samp2 = np.zeros(self.M)
        xnorm = np.sqrt( xnormMat )

        for j in range(self.numC):
          samp1 =  ( self.p + ( ( stocX  -self.p ) / xnorm ) * ( j / self.numC ) )
          samp2 = ( self.p - ( ( stocX - self.p ) / xnorm ) * ( j / self.numC ) ) 
          self.samples[ 2 * i * self.numC + 2 * j ]= samp1.T
          self.samples[ 2 * i * self.numC + 2 * j + 1 ] = samp2.T

    def getCells(self,):
      cInequality = np.zeros( (self.N,self.M,self.N) )
      for i in range(2):
        cInequality[i] = self.game.LossMatrix[i,...] - self.game.LossMatrix 
      return cInequality

    def chooseActionBpm(self, t):
      self.populateSamples(t)
      currentActiveActions = []
      activeActionsVec = []
      numActiveActions = len(activeActionsVec)
      score = np.zeros(self.N)
      for i in range(self.N):
        temp = self.cInequality[i] * self.samples.T
        boolMat =  temp <= np.zeros( (self.N, 2 * self.numC * self.sample_size) ) 
        sumBool = np.sum( boolMat, 0)
        if np.max(sumBool) == self.N:
          currentActiveActions.append(i)

      self.activeActions = currentActiveActions
      numActiveActions = len( currentActiveActions )
      for i in numActiveActions:
        score[  i ] = self.horizon - self.n[ i ]
      chosen = np.argmax(score)
      return chosen

    def myInverse(self, mat):

      if mat.shape == (1,1 ):
        inv =   1/ mat
      else:
        inv = np.linalg.inv(mat)
      return inv

    def update(self, action, feedback, outcome):
      self.n[action] += 1
      curr_sigmaInv = self.sigmaInv

      self.sigmaInv = curr_sigmaInv + self.Ps[action]
      self.sigma = np.linalg.inv(  self.sigmaInv )

      current_p = self.p
      Y_t = self.SignalMatrices[action] @ np.eye(self.M)[outcome]
      new_p = self.sigma @ ( curr_sigmaInv @ current_p + self.SignalMatrices[action].T @ self.sstInvs[action] @ Y_t  )
      new_p = abs(new_p)
      self.p = new_p/sum(new_p)




# class BPMPolicy : public Policy{
#   const MatrixXd lossMatrix;
#   const MatrixXi feedbackMatrix;
#   const std::vector<MatrixXd> signalMatrices; //note that each mat is AxM (not \sigma_i x M) for the ease of implementation
#   std::vector<MatrixXd> sstInvs; //myInverse( (s_i * (s_i.transpose())) )
#   const uint N;
#   const uint M;
#   const uint A; //number of alphabet (alphabet should be in 0,...,A-1)
#   const bool LEAST; //BPM-LEAST
#   const bool TS; //BPM-TS
#   const uint SAMPLE_NUM = 100; //100
#   VectorXd p_t;
#   MatrixXd sigmaInv_t;
#   MatrixXi feedback; //feedback[i][a] -> num of feedback a for action i
# public:
#   BPMPolicy(MatrixXd lossMatrix,
#      MatrixXi feedbackMatrix, uint N, uint M, uint A, uint mode, VectorXd p0, MatrixXd sigmaInv0):
#      lossMatrix(lossMatrix), feedbackMatrix(feedbackMatrix),
#      signalMatrices(calculateSignalMatrices(feedbackMatrix, N, M, A)) ,N(N), M(M), A(A), 
#      LEAST(mode==0), TS(mode==1), p_t(p0), sigmaInv_t(sigmaInv0) { //, alpha(alpha) {
#     if( (N <= 1) || (M <= 1) ){
#       std::cerr << "Error: trivial game" << std::endl;
#       exit(0);
#     }
#     if(! (LEAST||TS)){
#       std::cerr << "Error: unknown mode in policy_bpm" << std::endl;
#       exit(0);
#     }
#     if(p_t.size() != M){
#       std::cerr << "Error: p_t size is not M" << std::endl;
#       exit(0);
#     }
#     if( (sigmaInv0.rows() != M) || (sigmaInv0.cols() != M) ){
#       std::cerr << "Error: sigmaInv size is not (M, M)" << std::endl;
#       exit(0);
#     }
#     for(uint i=0;i<N;++i){
#       const MatrixXd s = signalMatrices[i];
#       const MatrixXd sstInv = myInverse( (s * (s.transpose())) );
# //      std::cout << "s=" << std::endl << s << std::endl;
# //      std::cout << "sstInv=" << std::endl << sstInv << std::endl;
#       sstInvs.push_back(sstInv);
#     }
#     feedback = MatrixXi::Zero(N, A);
#   }

#   # //obtain optimal action based on estimated p
#   # uint getOptimalAction(const VectorXd &p){
#   #   VectorXd expectedLoss = lossMatrix * p;
#   #   return vectorXdMinIndex(expectedLoss);
#   # }
#   VectorXd sampleP() const { //sample p from Gaussian. note that this p does not necessarily in simplex
#     # std::normal_distribution<> normal(0.0, 1.0); 
#     # VectorXd z = VectorXd::Zero(M);
#     # for(uint i=0;i<M;++i){
#     #   z[i] = normal(randomEngine);
#     # }
#     //create A
#     MatrixXd sigma = sigmaInv_t.inverse();
#     LLT<MatrixXd> lltOfSigma(sigma);
#     MatrixXd A = lltOfSigma.matrixL();
# //    std::cout << "Sigma=" << std::endl << sigma << std::endl;
# //    std::cout << "AAt=" << std::endl << A * A.transpose() << std::endl;
#     //mu+A*z
#     VectorXd r(p_t);
#     r += A*z;
#     return r;
#   }
#   bool nonNegative(const VectorXd &p){
#     for(uint i=0;i<M;++i){
#       if(p(i) < 0) return false;
#     }
#     return true;
#   }
#   virtual uint selectNextAction(uint t){
#     if(LEAST){
#       std::set<uint> optimalActions;
#       uint c=0;
#       while(c<SAMPLE_NUM){
#         VectorXd p = sampleP();
#         if(nonNegative(p)){
#           uint optimal = getOptimalAction(p);
#           optimalActions.insert(optimal);
#           c++;
#         }
#       }
#       std::set<uint>::iterator it;
#       uint minFeedback = INT_MAX;
#       uint minFeedbackOptimal = -1;
#       VectorXi feedbackRowwise = feedback.rowwise().sum();
#       for(it = optimalActions.begin(); it != optimalActions.end(); ++it){
#         if((*it) >= N){
#           std::cerr << "Error: optimal action not in [N] in bpm" << std::endl;exit(0);
#         }
#         if((uint)feedbackRowwise[*it] < minFeedback){
#           //std::cout << "*it=" << *it << std::endl;
#           minFeedback = feedbackRowwise(*it);
#           minFeedbackOptimal = *it;
#         }
#       }
#       if(minFeedbackOptimal == -1){
#         std::cerr << "Error: unexpected value in finding optimal action in bpm" << std::endl;exit(0);
#       }
#       return minFeedbackOptimal;
#     }else if(TS){
#       while(true){
#         VectorXd p = sampleP();
#         if(nonNegative(p)){
# //          std::cout << "sampled p=" << p << std::endl;
# //          std::cout << "p_t = " << p_t << std::endl;
# //          std::cout << "sigmaInv_t = " << sigmaInv_t << std::endl;
#           return getOptimalAction(p);
#         }
#       }
#     }
#     //return i;
#   }
#   //ignore no signal line
#   // ex.
#   // 1 0 -> 1 0
#   // 0 0    0 0 
#   MatrixXd myInverse(const MatrixXd &mat){
#     const uint s = mat.cols();
#     if(mat.cols() != mat.rows()){
#       std::cerr << "Error: matrix rows()!=cols()" << std::endl;
#       exit(0);
#     }
#     std::vector<uint> filledIndices;
#     for(uint i=0;i<s;++i){
#       bool blank = true;
#       for(uint j=0;j<s;++j){
#         if(mat(i,j)!=0){
#           blank = false;
#           break;
#         }
#       }
#       if(!blank){
#         filledIndices.push_back(i);
#       } 
#     }
#     const uint ms = filledIndices.size();
#     MatrixXd smlMat = MatrixXd::Zero(ms, ms);
#     for(uint i=0;i<ms;++i){
#       for(uint j=0;j<ms;++j){
#         smlMat(i,j) = mat(filledIndices[i], filledIndices[j]);
#       }
#     }
# //    std::cout << "smlMat=" << smlMat << std::endl;
#     MatrixXd smlInvMat = smlMat.inverse();
#     MatrixXd invMat = MatrixXd::Zero(s,s);
#     for(uint i=0;i<ms;++i){
#       for(uint j=0;j<ms;++j){
#         invMat(filledIndices[i], filledIndices[j]) = smlInvMat(i,j);
#       }
#     }
#     return invMat;
#   }
#   virtual void updateState(uint i, uint a, uint t){
#     feedback(i,a) += 1;
#     VectorXd feedbackVect = VectorXd::Zero(A);
#     feedbackVect(a) = 1;
#     const MatrixXd s = signalMatrices[i];
# //    std::cout << "s = " << s << std::endl;
#     const MatrixXd sstInv = sstInvs[i];
# //    std::cout << "sstInv = " << sstInv << std::endl;
#     const MatrixXd P_i = s.transpose()*sstInv*s;
#     const MatrixXd new_sigmaInv_t = sigmaInv_t + P_i;
#     const MatrixXd new_sigma_t = new_sigmaInv_t.inverse();
#     const VectorXd new_p_t = new_sigma_t * (sigmaInv_t*p_t + s.transpose()*sstInv*feedbackVect);
#     if(new_p_t.size() != M){
#       std::cerr << "Error: p_t size not M after update in bpm" << std::endl; exit(0);
#     }
#     if( (new_sigmaInv_t.rows() != M) || (new_sigmaInv_t.cols() != M) ){
#       std::cerr << "Error: sigmaInv size is not (M, M) after update in bpm" << std::endl; exit(0);
#     }
#     sigmaInv_t = new_sigmaInv_t;
#     p_t = new_p_t;
# //    std::cout << "Sigma_t =" << new_sigma_t << std::endl;
# //    std::cout << "p_t =" << new_p_t << std::endl;
#   }
#   virtual std::string toString(){
#     std::string str="BPM";
#     if(LEAST){
#       str += "-LEAST";
#     }else if(TS){
#       str += "-TS";
#     }
#     return str;
#   }
# };
