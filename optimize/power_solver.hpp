#pragma once
#include <random>
#include <chrono>
#include <memory>
#include "LLPack/utils/candy.hpp"
#include "LLPack/algorithms/random.hpp"
#include "LLPack/algorithms/algebra.hpp"
#include "../data/Bipartite.hpp"



namespace cat_tree {
  class PowerSolver {
    // parameters:
    int numL;
    int numU;
    const double *P;
    const Bipartite *m_to_l;
    double *q;
    
    // internal
    int N;
    int K;
    std::mt19937 rng;

  public:
    // options
    struct Options
    {
      int powerMaxIter;
      double powerConverge;
      int maxSubspaceDim;
      double significant;
      int maxFail;
      /* ---------- */
      int maxIter; // iteration number
      double shrinkRatio; // Shrinking Ratio for line search
      Options()
      {
        powerMaxIter = 200;
        powerConverge = 1e-5;
        maxSubspaceDim = 100;
        significant = 0.01;
        maxFail = 10;
        /* ---------- */
        maxIter = 20;
        shrinkRatio = 0.8;
      }
    } options;



  public:

    PowerSolver() {}


    
  private:

    void initY( double *y )
    {
      for ( int n=0; n<N; n++ ) {
        std::vector<double> tmp = rndgen::rnd_unit_vec<double>( K, rng );
        algebra::watershed( &tmp[0], y + n * K, K );
      }
    }
    


    inline void enforce_y( double *y )
    {
      algebra::copy( y, P, numL * K );
    }

    inline void q_from_y( double *q, double *y )
    {
      // q(l) = sum_n beta_(n,l) * y(n)
      // beta(n,l) = alpha(n,l) / [ sum_n alpha(n,l) ]
      for ( int l=0; l<m_to_l->sizeB(); l++ ) {



        auto _to_n = m_to_l->getFromSet(l);
        if ( 0 == _to_n.size() ) continue;
        
        algebra::zero( q + l * K, K );
        double s = 0.0;
        for ( auto& ele : _to_n ) {
          const int &n = ele.first;
          const double &alpha = ele.second;
          s += alpha;
          algebra::addScaledTo( q + l * K, y + n * K, K, alpha );
        }
        s = 1.0 / s;
        for ( int k=0; k<K; k++ ) {
          q[l*K+k] *= s;
        }

      }
    }

    inline void y_from_q( double *y, double *q, bool debug=false )
    {
      double tmp[K];

      // y(n) = sum_l alpha(n,l) q(l)
      for ( int n=0; n<N; n++ ) {
        auto _to_l = m_to_l->getToSet(n);
        algebra::copy( tmp, y + n * K, K );
        algebra::zero( y + n * K, K );
        for ( auto& ele : _to_l ) {
          const int &l = ele.first;
          const double &alpha = ele.second;
          algebra::addScaledTo( y + n * K, q + l * K, K, alpha );
        }
        if ( numL <= n ) {

          if ( debug ) {
            DebugInfo( "before:" );
            printVec( tmp, K );
            DebugInfo( "after:" );
            printVec( y + n * K, K );
            ResumeOnRet();
          }

        }
      }

    }
    
    inline std::unique_ptr<double> PowerIter( double &energy )
    {
      std::unique_ptr<double> owner_y( new double[K*N] );
      double *y = owner_y.get();
      y_from_q( y, q );
      enforce_y( y );

      double lastEnergy = - 1.0;

      for ( int iter=0; iter<options.powerMaxIter; iter++ ) {
        
        q_from_y( q, y );

        

        y_from_q( y, q, false );

        // debugging:
        // DebugInfo( "estimation of y: " );  
        // for ( int n=0; n<numL; n++ ) {
        //   if ( 0.5 < P[n*K+9] ) {
        //     algebra::emphasizeDim( y + n * K, K, 9 );
        //   }
        // }
        // ResumeOnRet();

        enforce_y( y );
        
        energy = 0.0;
        for ( int n=0; n<N; n++ ) {
          auto& _to_l = m_to_l->getToSet( n );
          for ( auto& ele : _to_l ) {
            int l = ele.first;
            double alpha = ele.second;
            energy += alpha * algebra::dist2( y + n * K, q + l * K, K );
          }
        }

        Info( "Iter: %d, Energy: %.6lf", iter, energy );

        if ( lastEnergy > 0.0 && fabs( lastEnergy - energy ) < options.powerConverge ) {
          break;
        }
        lastEnergy = energy;
      }
      
      return owner_y;
    }


  public:
    double operator()( int numL1, int numU1, const double* P1, 
                       const Bipartite *m_to_l1, double *q1 )
    {
      // initialize
      numL = numL1;
      numU = numU1;
      N = numL + numU;
      P = P1;
      m_to_l = m_to_l1;
      q = q1;
      K = LabelSet::classes;
      

      rng.seed( time( NULL ) );
      
      double energy;
      std::unique_ptr<double> y = PowerIter( energy );
      q_from_y( q, y.get() );
      return energy;
      
    }
    
    
  };
}
