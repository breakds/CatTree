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
    int B; // num of basis
    std::vector<std::unique_ptr<double> > Y;
    std::vector<std::unique_ptr<double> > E;
    std::mt19937 rng;

  public:
    // options
    struct Options
    {
      int powerMaxIter;
      int powerConverge;
      int maxSubspaceDim;
      int significant;
      int maxFail;
      /* ---------- */
      int maxIter; // iteration number
      double shrinkRatio; // Shrinking Ratio for line search
      Options()
      {
        powerMaxIter = 100;
        powerConverge = 0.01;
        maxSubspaceDim = 1000;
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
        memcpy( y + n * K, &tmp[0], K * sizeof(double) );
        double s = algebra::sum_vec( y + n * K, K );
        algebra::scale( y + n * K, K, 1.0 / s );
      }
    }
    


    inline void q_from_y( double *q, double *y )
    {
      // q(l) = sum_n beta_(n,l) * y(n)
      // beta(n,l) = alpha(n,l) / [ sum_n alpha(n,l) ]
      for ( int l=0; l<m_to_l->sizeB(); l++ ) {
        auto _to_n = m_to_l->getFromSet(l);
        if ( 0 == _to_n.size() ) {
          Error( "leaf not touched!" );
          exit( -1 );
        }
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

    inline double y_from_q( double *y, double *q )
    {
      double tmp[K];
      double energy = 0.0;
      
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
        energy += algebra::dist2( y + n * K, tmp, K );
      }
      
      return energy;
    }
    
    inline std::unique_ptr<double> PowerIter()
    {
      std::unique_ptr<double> owner_y( new double[K*N] );
      double *y = owner_y.get();
      initY( y );
      
      Info( "========== PowerIter ==========" );
      for ( int iter=0; iter<options.powerMaxIter; iter++ ) {

        q_from_y( q, y );
        
        double energy = y_from_q( y, q );
        Info( "energy: %.5lf\n", energy );
        if ( sqrt(energy) < options.powerConverge ) {
          break;
        }
      }

      return owner_y;
    }


    void InitBasis()
    {
      B = 0;
      int failed = 0;
      while ( B < options.maxSubspaceDim ) {
        std::unique_ptr<double> y_new = PowerIter();
        std::unique_ptr<double> tmp( new double( K * N ) );
        algebra::copy( tmp.get(), y_new.get(), K * N );
        for ( auto& e : E ) {
          double coef = algebra::dotprod( e.get(), tmp.get(), K * N );
          algebra::minusScaledFrom( tmp.get(), e.get(), K * N, coef );
        }
        if ( options.significant < algebra::norm_l1( tmp.get(), K * N ) ) {
          algebra::normalize_vec( tmp.get(), tmp.get(), K * N );
          E.push_back( std::move( tmp ) );
          Y.push_back( std::move( y_new ) );
          B++;
          failed = 0;
        } else {
          failed++;
          if ( failed > options.maxFail ) {
            break;
          }
        }
      }
    }


    double quadEnergy( const double *x )
    {
      double energy = 0.0;
      for ( int d=0; d<K*numL; d++ ) {
        double tmp = P[d];
        for ( int b=0; b<B; b++ ) {
          tmp -= x[b] * Y[b].get()[d];
        }
        energy += tmp * tmp;
      }
      energy *= 0.5;
      return energy;
    }


    void quadOpt()
    {
      double *YTP = new double[B];
      double *YTY = new double[B*B]; // YTY is row major!!!

      for ( int b=0; b<B; b++ ) {
        YTP[b] = algebra::dotprod( Y[b].get(), P, K * numL );
        for ( int c=0; c<B; c++ ) {
          YTY[ c * B + b ] = algebra::dotprod( Y[b].get(), Y[c].get(), K * numL );
        }
      }


      double x[B];
      for ( int b=0; b<B; b++ ) x[b] = 1.0 / B;
      double energy = quadEnergy( x );
      
      double g[B];

      for ( int iter=0; iter<options.maxIter; iter++ ) {
        // negative gradient
        // YTP - YTY * x
        for ( int b=0; b<B; b++ ) {
          g[b] = YTP[b] - algebra::dotprod( YTY + b * B, x, B );
        }

        // line search
        double newEnergy = 0.0;
        double stepSize = 1.0;
        double newX[B];
        double t[B];
        do {
          algebra::copy( t, x, B );
          algebra::minusScaledFrom( t, g, B, stepSize );
          algebra::watershed( t, newX, B );
          newEnergy = quadEnergy( newX );
          stepSize *= options.shrinkRatio;
        } while ( newEnergy > energy );

        energy = newEnergy;
        copy( x, newX, B );
        
      }

      
      double *final_y = new double[K*N];
      for ( int d=0; d<K*N; d++ ) {
        final_y[d] = 0.0;
        for ( int b=0; b<B; b++ ) {
          final_y[d] += x[b] * Y[b].get()[d];
        }
      }
      q_from_y( q, final_y );
      
      DeleteToNullWithTestArray( final_y );
      DeleteToNullWithTestArray( YTP );
      DeleteToNullWithTestArray( YTY );
    }
    
    
  public:
    void operator()( int numL1, int numU1, const double* P1, 
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
      

      E.clear();
      Y.clear();
      rng.seed( std::chrono::system_clock::now().time_since_epoch().count() );

      
      
      InitBasis();
      // quadOpt();

      Info( "%d Basis found.", B );
      
    }
    
    
  };
}
