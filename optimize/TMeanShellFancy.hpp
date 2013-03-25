#pragma once

#include <omp.h>
#include <vector>
#include <memory>
#include "RanForest/RanForest.hpp"
#include "LLPack/algorithms/heap.hpp"
#include "LLPack/algorithms/algebra.hpp"
#include "proj_grad.hpp"
#include "../data/vector.hpp"


using ran_forest::Bipartite;

namespace cat_tree {

  template <typename dataType = float>
  class TMeanShell
  {
  public:
    // std::vector<dataType> centers;
    std::vector<std::unique_ptr<dataType> > centers;

    struct Options
    {
      int maxIter;
      int replicate;
      double converge;
      Options() : maxIter(100), replicate(10), converge(1e-5) {}
    } options;

    struct optStorage {
      std::vector<std::vector<double> > D;
      optStorage() : D() {}
    };

    struct alphaStorage {
      std::vector<double> D;
    };


    TMeanShell() : centers(), options() {}

    template <typename feature_t, template <typename T = feature_t, typename... restArgs> class container>
    void Clustering( const container<feature_t> &feat,
                     int dim,
                     Bipartite& n_to_l )
    {
      int N_int = n_to_l.sizeA();
      size_t N = n_to_l.sizeA();
      size_t L = n_to_l.sizeB();
      
      
      // centers.resize( L );
      // for ( auto& ele : centers ) {
      //   ele.reset( new dataType[dim] );
      // }

      

      Bipartite bimap( N, L );
      
      // initialization
      ProjGradSolver<optStorage> solveCenters( L * dim, 
                                               [&dim,&N,&L,&n_to_l,&feat]
                                               ( std::vector<double>& x, optStorage& store ) 
                                               {
                                                 for ( size_t l=0; l<L; l++ ) {
                                                   auto& _to_n = n_to_l.to( l );
                                                   if ( 0 < _to_n.size() ) {
                                                     double s = 0.0;
                                                     for ( auto& ele : _to_n ) {
                                                       const size_t n = static_cast<size_t>( ele.first );
                                                       const double& alpha = ele.second;
                                                       for ( int j=0; j<dim; j++ ) {
                                                         x[l*dim+j] += alpha * feat[n][j];
                                                       }
                                                       s += alpha;
                                                     }
                                                     scale( &x[l*dim], dim, 1.0 / s );
                                                   }
                                                 }
                                                 // D(n) = x(n) - \sum alpha(n,l) * C(l)
                                                 store.D.resize( N );
                                                 for ( size_t n = 0; n < N; n ++ ) {
                                                   store.D[n].resize( dim );
                                                 }
                                               } );
      solveCenters.options.converge = 100.0;
      
      double last_e = 0.0;
      
      for ( int iter=0; iter<options.maxIter; iter++ ) {
        
        Info( "TMeans (fancy) iter %d", iter );
        
        Bipartite& graph = ( 0 == iter ) ? n_to_l : bimap;
        
        // update centers
        Info( "Updating Centers ..." );
        auto e = solveCenters( [&dim,&N,&L,&graph,&feat] // energy
                               ( const std::vector<double>& x, optStorage& store ) 
                               {
                                 // update D also
                                 double e = 0.0;

#                                pragma omp parallel for
                                 for ( size_t n=0; n<N; n++ ) {
                                   double *tmp = &store.D[n][0];
                                   auto& _to_l = graph.from( n );
                                   for ( int j=0; j<dim; j++ ) {
                                     tmp[j] = feat[n][j];
                                   }
                                   for ( auto& ele : _to_l ) {
                                     const size_t l = static_cast<size_t>( ele.first );
                                     const double& alpha = ele.second;
                                     algebra::minusScaledFrom( tmp, &x[l*dim], dim, alpha );
                                   }
                                   e += norm2( tmp, dim );
                                 }
                                 return e;
                               }, 

                               [&dim,&N,&L,&graph,&feat] // negative derivative
                               ( std::vector<double>& d, const std::vector<double> __attribute__((__unused__)) &x, 
                                 optStorage& store )
                               {
                                 // D(n) = x(n) - \sum_l alpha(n,l) * C(l) 
                                 // - deriv(l) = \sum_n [ D(n) * alpha(n,l) ]
                                 double *tmp = &d[0];
                                 for ( size_t l=0; l<L; l++ ) {
                                   algebra::zero( tmp, dim );
                                   auto& _to_n = graph.to( l );
                                   for ( auto& ele : _to_n ) {
                                     const size_t n = static_cast<size_t>( ele.first );
                                     const double& alpha = ele.second;
                                     addScaledTo( tmp, &store.D[n][0], dim, alpha );
                                   }
                                   tmp += dim;
                                 }
                               },

                               [&dim,&N,&L,&graph,&feat] // projection
                               ( std::vector<double>& x, optStorage __attribute__((__unused__)) &store )
                               {
                                 double *c = &x[0];
                                 double tmp[dim];
                                 for ( size_t l=0; l<L; l++ ) {
                                   algebra::copy( tmp, c, dim );
                                   algebra::watershed( tmp, c, dim );
                                 }
                               } );

        
        if ( 0 < iter && fabs( e - last_e ) < options.converge ) {
          break;
        } else {
          last_e = e;
        }


        // update alphas
        Info( "Updating Graph ..." );
        bimap.clear();
        

        ProgressBar alphaProgress( N_int );
        for ( int n=0; n<N_int; n++ ) {
          auto& _to_l = n_to_l.from( n );
          int supportSize = _to_l.size();
          ProjGradSolver<alphaStorage> solveAlphas( supportSize, 
                                                    [&dim,&N,&L,&_to_l,&feat]
                                                    ( std::vector<double>& alphas, alphaStorage& store ) 
                                                    {
                                                      double wt = 1.0 / _to_l.size();
                                                      for ( auto& ele : alphas ) {
                                                        ele = wt;
                                                      }
                                                      // D = x(n) - \sum alpha(n,l) * C(l)
                                                      store.D.resize( dim );
                                                    } );
          
          solveAlphas.options.iterEnergyInfo = false;
          
          solveAlphas( [&dim,&N,&L,&_to_l,&n,&feat,&solveCenters,&supportSize] // energy
                       ( const std::vector<double>& alphas, alphaStorage& store ) 
                       {
                         double *D = &store.D[0];
                         for ( int j=0; j<dim; j++ ) {
                           D[j] = feat[n][j];
                         }
                         for ( int i=0; i<supportSize; i++ ) {
                           size_t l = static_cast<size_t>( _to_l[i].first );
                           // D -= alpha(i) * C(l)
                           algebra::minusScaledFrom( D, &solveCenters.x[ l * dim ], dim, alphas[i] );
                         }
                         return algebra::norm2( D, dim );
                       },
                       
                       [&dim,&N,&L,&_to_l,&n,&feat,&solveCenters,&supportSize] // negative derivative
                       ( std::vector<double>& d, const std::vector<double> __attribute__ ((__unused__)) &alphas, 
                         alphaStorage& store )
                       {
                         // -deriv(i) = C(l)^T D(i)
                         for ( int i=0; i<supportSize; i++ ) {
                           size_t l = static_cast<size_t>( _to_l[i].first );
                           d[i] = algebra::dotprod( &solveCenters.x[l*dim], &store.D[0], dim );
                         }
                       },

                       [&dim,&N,&L,&_to_l,&n,&feat,&solveCenters,&supportSize] // projection
                       ( std::vector<double>& alphas, alphaStorage __attribute__ ((__unused__)) &store )
                       {
                         double tmp[supportSize];
                         algebra::copy( tmp, &alphas[0], supportSize );
                         algebra::watershed( tmp, &alphas[0], supportSize );
                       } );

          heap<double,int> ranker( options.replicate );
          for ( int i=0; i<supportSize; i++ ) {
            ranker.add( solveAlphas.x[i], i );
          }


          // post projection
          double alphas[ranker.len];
          for ( int j=0; j<ranker.len; j++ ) {
            alphas[j] = solveAlphas.x[ranker[j]];
          }

          {
            double tmp[ranker.len];
            algebra::copy( tmp, alphas, supportSize );
            algebra::watershed( tmp, alphas, supportSize );
          }

          // update bimap
          for ( int j=0; j<ranker.len; j++ ) {
            bimap.add( n, _to_l[ranker[j]].first, alphas[j] );
          }
          
          alphaProgress.update( n + 1, "updating graph" );
        } // end for n
        
      }
      
      n_to_l = std::move( bimap );
    }
    
  };
}
