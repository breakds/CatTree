#pragma once


namespace cat_tree
{
  class Solver {
    /* Solver for objective function
     *           \sum_m      ( \sum_l alpha(m,l) * q(l) - P(m) )^2
     *  + \beta  \sum_{i,j}  w(i,j) [ \sum_l (alpha(i,l)-alpha(j,l)) q(l) ]^2
     *
     *  where
     *  1. P(m)           -    the groud truth label distribution for patch m
     *  2. alpha(m,l)     -    coefficients for voter l against patch m
     *  3. q(l)           -    the self-voting label distribution of voter l
     *  4. beta           -    coefficient for the regularization term
     *  5. w(i,j)         -    feature similarity between patch i and patch j
     *
     *  and
     *  1. D(m)           =    \sum_l alpha(m,l) * q(l) - P(m)
     *  2. d(n)           =    \sum_l (alpha(i_n,l) - alpha(j_n,l)) q(l)
     *  2. numL           -    number of labeled patches
     *  3. numU           -    number of unlabeled patches
     *  4. K              -    number of classes
     *  5. L              -    number of voters
     *  6. N              -    number of patch neighbor pairs
     */
  public:
    struct Options
    {
      double beta; // coefficient for the pairwise regularization term
      int maxIter; // iteration number
      double shrinkRatio; // Shrinking Ratio for line search
      double wolf; // Sufficent descreasing condition coefficient

      Options()
      {
        beta = 1.0;
        maxIter = 10;
        shrinkRatio = 0.8;
        wolf = 0.999;
      }
    } options;

  private:

    const double *P;
    const double *w;

  
    double *q;

    double *D;
    double *d;
  
    int K, L, N, numL, numU;
  
    const Bipartite *m_to_l;
    const Bipartite *pair_to_l;
  
    const std::vector<std::pair<int,int> > *patchPairs;


  public:
  
    // constructor
    Solver()
    {
      D = nullptr;
      P = nullptr;
      q = nullptr;
      d = nullptr;
      w = nullptr;
    
      K = 0;
      L = 0;
      N = 0;
      numL = 0;
      numU = 0;
    
      m_to_l = nullptr;
      pair_to_l = nullptr;
      patchPairs = nullptr;
    }
  
  private:
  
  

    /* ---------- D(m) ---------- */
    // D(m) = sum alpha(l,m)*q(l,m) - P(m)
    // logic checked
    inline void update_D( int m )
    {
      auto& _to_l = m_to_l->getToSet( m );

      // Set D(m) to 0
      double *t = D + m * K;
      zero( t, K );

      // Note t is an alias of D(m) now
    
      // add q(l) * alpha(m,l) to D(m)
      for ( auto& ele : _to_l ) {
        int l = ele.first;
        double alpha = ele.second;
        addScaledTo( t, q + l * K, K, alpha );
      }

      // minus P(m) from D(m)
      minusFrom( t, P + m * K, K );
    }
  
  
    inline void update_D( int m, int l, const double *q_l, double alpha )
    {
      // D(m) = D(m) - q(l) * alpha(m,l)
      minusScaledFrom( D + m * K, q + l * K, K, alpha );
      // D(m) = D(m) + q_new(l) * alpha(m,l)
      addScaledTo( D + m * K, q_l, K, alpha );
    }

    inline void altered_D( int m, int l, const double *q_l, double alpha, double *dst )
    {
      copy( dst, D + m * K, K );
      // D(m) = D(m) - q(l) * alpha(m,l)
      minusScaledFrom( dst, q + l * K, K, alpha );
      // D(m) = D(m) + q_new(l) * alpha(m,l)
      addScaledTo( dst, q_l, K, alpha );
    }




    /* ---------- d(n) ---------- */

    // d(n) = sum_l (alpha(i_n,l) - alpha(j_n,l)) * q(l)
    // logic checked
    inline void update_d( int n )
    {
      // alias t <- d(n)
      double *t = d + n * K;
      zero( t, K );
      
      { 
        // handle i's side
        int i = (*patchPairs)[n].first;
        auto& _to_l = m_to_l->getToSet( i );
        for ( auto& ele : _to_l ) {
          int l = ele.first;
          double alpha = ele.second;
          addScaledTo( t, q + l * K, K, alpha );
        }
      }

      {
        // handle j's side
        int j = (*patchPairs)[n].second;
        auto& _to_l = m_to_l->getToSet( j );
        for ( auto& ele : _to_l ) {
          int l = ele.first;
          double alpha = ele.second;
          minusScaledFrom( t, q + l * K, K, alpha );
        }
      }
    }

    inline void update_d( int n, int l, const double *q_l, double alpha_i, double alpha_j )
    {
      double *t = d + n * K;
      // d(n) = d(n) - alpha(i,l) * q(l) + alpha(i,l) * q_l
      minusScaledFrom( t, q + l * K, K, alpha_i );
      addScaledTo( t, q_l, K, alpha_i );
    
      // d(n) = d(n) + alpha(j,l) * q(l) - alpha(j,l) * q_l
      addScaledTo( t, q + l * K, K, alpha_j );
      minusScaledFrom( t, q_l, K, alpha_j );
    }

    inline void altered_d( int n, int l, const double *q_l, double alpha_i, double alpha_j, double *dst )
    {
      copy( dst, d + n * K, K );
    
      // d(n) = d(n) - alpha(i,l) * q(l) + alpha(i,l) * q_l
      minusScaledFrom( dst, q + l * K, K, alpha_i );
      addScaledTo( dst, q_l, K, alpha_i );
    
      // d(n) = d(n) + alpha(j,l) * q(l) - alpha(j,l) * q_l
      addScaledTo( dst, q + l * K, K, alpha_j );
      minusScaledFrom( dst, q_l, K, alpha_j );
    }

  

  
    /* ---------- Energy Computation ---------- */

    // logic checked
    inline double total_energy()
    {

      double energy_first = 0.0;

      // sum_m D(m)^2
      double *Dp = D;
      for ( int m=0; m<numU; m++ ) {
        energy_first += norm2( Dp, K );
        Dp += K;
      }

      double energy_second = 0.0;

      // sum_n d(n)^2 * w(n)
      // where w(n) = w(i_n,j_n)
      double *dp = d;
      for ( int n=0; n<N; n++ ) {
        energy_second += norm2( dp, K ) * w[n];
        dp += K;
      }

      // debugging:
      printf( "%.6lf vs. %.6lf\n", energy_first, energy_second );
    
      return energy_first + energy_second * options.beta;
    }

    /* Restricted Energy on q(l):
     * sum_{m \in l} D(m)^2 + \beta \sum_{n \in l} w(i_n,j_n) * d(n)^2
     */
    // logic checked
    inline double restrict_energy( int l )
    {

      auto& _to_m = m_to_l->getFromSet( l );
      auto& _to_n = pair_to_l->getFromSet( l );
    
    
      // energy_first = sum_{m \in l} D(m)^2
      double energy_first = 0.0;
      for ( auto& ele : _to_m ) {
        int m = ele.first;
        energy_first += norm2( D + m * K, K );
      }
    

      // energy_second = \sum_{n \in l} w(i_n,j_n) * d(n)^2
      double energy_second = 0.0;
      for ( auto& ele : _to_n ) {
        int n = ele.first;
        energy_second += norm2( d + n * K, K ) * w[n];
      }

      return energy_first + energy_second * options.beta;
    }

  
    /* Restricted Energy on q(l) replaced with q_l
     * sum_{m \in l} D(m)^2 + \beta \sum_{n \in l} w(i_n,j_n) * d(n)^2
     */
    
    inline double restrict_energy( int l, const double *q_l )
    {
      auto& _to_m = m_to_l->getFromSet( l );
      auto& _to_n = pair_to_l->getFromSet( l );

      std::unordered_map<int,double> alphas;

      double t[K];

      // energy_first = sum_{m \in l} altered_D(m)^2
      double energy_first = 0.0;
      for ( auto& ele : _to_m ) {
        int m = ele.first;
        double alpha = ele.second;
        // map alpha and m
        alphas[m] = alpha;
        altered_D( m, l, q_l, alpha, t );
        energy_first += norm2( t, K );
      }
    

      // energy_second = \sum_{n \in l} w(i_n,j_n) * d(n)^2
      double energy_second = 0.0;
      for ( auto& ele : _to_n ) {
        int n = ele.first;
        int i = (*patchPairs)[n].first;
        int j = (*patchPairs)[n].second;
        double alpha_i = 0.0;
        if ( alphas.end() != alphas.find(i) ) {
          alpha_i = alphas[i];
        }
        double alpha_j = 0.0;
        if ( alphas.end() != alphas.find(j) ) {
          alpha_j = alphas[j];
        }
        altered_d( n, l, q_l, alpha_i, alpha_j, t );
        energy_second += norm2( t, K ) * w[n];
      }

      return energy_first + energy_second * options.beta;
    }




    /* ---------- Restricted Derivative ---------- */

    // logic checked
    inline void restrict_deriv( int l, double *deriv )
    {

      zero( deriv, K );

      std::unordered_map<int,double> alphas;
    
      // deriv = sum_m alpha(l,m) * D(m)
      auto& _to_m = m_to_l->getFromSet( l );
      for ( auto& ele : _to_m ) {
        int m = ele.first;
        double alpha = ele.second;
        // map alpha and m
        alphas[m] = alpha;
        addScaledTo( deriv, D + m * K, K, alpha );
      }
    
      // deriv += beta * sum_n (alpha(i_n,l)-alpha(j_n,l)) * d(n) * w(n)
      auto& _to_n = pair_to_l->getFromSet( l );
      for ( auto& ele : _to_n ) {
        int n = ele.first;
        int i = (*patchPairs)[n].first;
        int j = (*patchPairs)[n].second;
        double alpha_i = 0.0;
        if ( alphas.end() != alphas.find(i) ) {
          alpha_i = alphas[i];
        }
        double alpha_j = 0.0;
        if ( alphas.end() != alphas.find(j) ) {
          alpha_j = alphas[j];
        }
        addScaledTo( deriv, d + n * K, K, options.beta * (alpha_i - alpha_j) * w[n] );
      }
    }


    /* ---------- update q ---------- */

    inline void update_q( int l )
    {
    
      double t0[K];
      double t1[K];
      double t2[K];

      // Get the negative derivative
      restrict_deriv( l, t0 );
      negate( t0, K );

    
      // Line Search Parabola
    
      double t3[K];        
      bool updated = false;
      double a = 1.0; // initial step size
      double dE2 = norm2( t0, K );
      if ( dE2 < 1e-6 ) {
        return;
      }
      double E0 = restrict_energy( l );

      memcpy( t1, q + l * K, sizeof(double) * K );
      addScaledTo( t1, t0, K, a );
      watershed( t1, t2, K );
      double E_a = restrict_energy( l, t2 );
    
      if ( E_a >= E0 ) {

        // Shrinking Branch
        for ( int i=0; i<40; i++ ) {

          a = ( a * a ) * dE2 / ( 2 * ( E_a - (E0 - dE2 * a ) ) );

          // update E_a
          memcpy( t1, q + l * K, sizeof(double) * K );
          addScaledTo( t1, t0, K, a );
          watershed( t1, t2, K );
          E_a = restrict_energy( l, t2 );

          if ( E_a < E0 ) {
            updated = true;
            break;
          }
        }

      } else { 
        // Expanding Branch
      
        updated = true;
      
        double E_best = E_a;

        for ( int i=0; i<40; i++ ) {
          if ( E_a > E0 - dE2 * a * ( 1 - 0.5 / 2.0 ) ) {
            double b = ( a * a ) * dE2 / ( 2 * ( E_a - (E0 - dE2 * a ) ) );
          
            // update E_a
            memcpy( t1, q + l * K, sizeof(double) * K );
            addScaledTo( t1, t0, K, b );
            watershed( t1, t3, K );
            E_a = restrict_energy( l, t3 );
    
            if ( E_a < E_best ) {
              E_best = E_a;
              memcpy( t2, t3, sizeof(double) * K );
            }
          } else {
            a = a * 2.0;
          }

          memcpy( t1, q + l * K, sizeof(double) * K );
          addScaledTo( t1, t0, K, a );
          watershed( t1, t3, K );
          E_a = restrict_energy( l, t3 );

          if ( E_a < E_best ) {
            E_best = E_a;
            memcpy( t2, t3, sizeof(double) * K );
          } else {
            break;
          }
        }

      }

      if ( updated ) {

        auto& _to_m = m_to_l->getFromSet( l );
        std::unordered_map<int,double> alphas;
        for ( auto& ele : _to_m ) {
          int m = ele.first;
          double alpha = ele.second;
          alphas[m] = alpha;
          update_D( m, l, t2, alpha );
        }

        auto& _to_n = pair_to_l->getFromSet( l );
        for ( auto& ele : _to_n ) {
          int n = ele.first;
          int i = (*patchPairs)[n].first;
          int j = (*patchPairs)[n].second;
          double alpha_i = 0.0;
          if ( alphas.end() != alphas.find(i) ) {
            alpha_i = alphas[i];
          }
          double alpha_j = 0.0;
          if ( alphas.end() != alphas.find(j) ) {
            alpha_j = alphas[j];
          }
          update_d( n, l, t2, alpha_i, alpha_j );
        }

        memcpy( q + l * K, t2, sizeof(double) * K );

      }
    
    }

  public:

    void operator()( int numL1, int numU1, int L1,
                     const Bipartite *m_to_l1,
                     const Bipartite *pair_to_l1,
                     const std::vector<std::pair<int,int> > *patchPairs1,
                     const double *w1,
                     const double *P1, double *q1 )
    {

      K = LabelSet::classes;
      L = L1;
      N = static_cast<int>( patchPairs1->size() );
      numL = numL1;
      numU = numU1;

      P = P1;
      w = w1;
      q = q1;
      d = new double[N*K];
      D = new double[(numL + numU) * K];

      pair_to_l = pair_to_l1;
      m_to_l = m_to_l1;
      patchPairs = patchPairs1;

    
      
      // Update D(m)'s
      for ( int m=0; m<numL; m++ ) {
        update_D(m);
      }

      // update d(n)'s
      for ( int n=0; n<N; n++ ) {
        update_d(n);
      }

    
      for (int iter=0; iter<options.maxIter; iter++ ) {

        // Update q(l)'s
        for ( int l=0; l<L; l++ ) {
          update_q(l);
        }

        Info( "Iteration %d - Energy: %.5lf\n", iter, total_energy() );
      
      }

      DeleteToNullWithTestArray( D );
      DeleteToNullWithTestArray( d );
    }


  };
}

