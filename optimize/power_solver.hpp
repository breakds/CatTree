#pragma once
#include <random>
#include <chrono>
#include <memory>
#include "LLPack/utils/candy.hpp"
#include "LLPack/algorithms/random.hpp"
#include "LLPack/algorithms/algebra.hpp"
#include "PatTk/data/Label.hpp"

using ran_forest::Bipartite;
using namespace PatTk;

namespace cat_tree {
  class PowerSolver {
    // parameters:
    size_t numL;
    size_t numU;
    const double *P;
    const Bipartite *m_to_l;
    double *q;
    std::vector<int> label;
    
    // internal
    size_t N;
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
        powerConverge = 1e-4;
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
      for ( size_t n=0; n<N; n++ ) {
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
#     pragma omp parallel for
      for ( size_t l=0; l<m_to_l->sizeB(); l++ ) {
        auto _to_n = m_to_l->to(l);
        if ( 0 == _to_n.size() ) continue;
        
        algebra::zero( q + l * K, K );
        double s = 0.0;
        for ( auto& ele : _to_n ) {
          const size_t &n = ele.first;
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

    inline void y_from_q( double *y, double *q )
    {

      // y(n) = sum_l alpha(n,l) q(l)
#     pragma omp parallel for
      for ( size_t n=0; n<N; n++ ) {
        double tmp[K];
        auto _to_l = m_to_l->from(n);
        algebra::copy( tmp, y + n * K, K );
        algebra::zero( y + n * K, K );
        for ( auto& ele : _to_l ) {
          const size_t &l = ele.first;
          const double &alpha = ele.second;
          algebra::addScaledTo( y + n * K, q + l * K, K, alpha );
        }
      }
    }
    
    inline double PowerIter( std::unique_ptr<double> &yptr )
    {
      if ( !yptr ) {
        yptr.reset( new double[K*N] );
        y_from_q( yptr.get(), q );
      }
      double *y = yptr.get();
      enforce_y( y );

      double lastEnergy = - 1.0;


      for ( int iter=0; iter<options.powerMaxIter; iter++ ) {
        
        q_from_y( q, y );

        y_from_q( y, q);

        enforce_y( y );
        
        double energy = 0.0;
#       pragma omp parallel for reduction(+:energy)
        for ( size_t n=0; n<N; n++ ) {
          auto& _to_l = m_to_l->from( n );
          for ( auto& ele : _to_l ) {
            size_t l = ele.first;
            double energySeg = ele.second * algebra::dist2( y + n * K, q + l * K, K );
            energy += energySeg;
          }
        }

        Info( "Iter: %d, Energy: %.6lf", iter, energy );

        if ( lastEnergy > 0.0 && fabs( lastEnergy - energy ) < options.powerConverge ) {
          return energy;
        }
        lastEnergy = energy;
      }
      return lastEnergy;

    }


  public:
    double operator()( size_t numL1, size_t numU1, const double* P1, 
                       const Bipartite *m_to_l1, double *q1, 
                       std::unique_ptr<double> &y )
    {
      // initialize
      numL = numL1;
      numU = numU1;
      N = numL + numU;
      P = P1;
      m_to_l = m_to_l1;
      q = q1;
      K = LabelSet::classes;
      label.resize( numL );
      for ( size_t i=0; i<numL; i++ ) {
        for ( int k=0; k<K; k++ ) {
          if ( P[i*K+k] > 0.9 ) {
            label[i] = k;
            break;
          }
        }
      }
      
      
      rng.seed( time( NULL ) );
      
      double energy = PowerIter( y );
      q_from_y( q, y.get() );
      return energy;
    }
  };
}
