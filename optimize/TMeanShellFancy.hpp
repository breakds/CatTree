#pragma once

#include <omp.h>
#include <vector>
#include <memory>
#include "RanForest/RanForest.hpp"
#include "LLPack/utils/time.hpp"
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
    std::vector<std::unique_ptr<double> > centers;

    struct Options
    {
      int maxIter;
      int replicate;
      double converge;
      double epsilon; // if the weight of an edge in the bipartite
                      // graph is lower than this, it will be ignored.
      Options() : maxIter(100), replicate(10), converge(1e-5), epsilon(1e-5) {}
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
      
      
      centers.resize( L );
      for ( auto& ele : centers ) {
        ele.reset( new double[dim] );
      }
      
      double last_e = 0.0;

      Bipartite bimap( N, L );

      for ( int iter=0; iter<options.maxIter; iter++ ) {
        
        Info( "TMeans (fancy) iter %d", iter );
        
        Bipartite& graph = ( 0 == iter ) ? n_to_l : bimap;
        
        // update centers
        Info( "Updating Centers ..." );

#       pragma parallel for
        for ( size_t l = 0; l < L; l ++ ) {
          zero( centers[l].get(), dim );
          auto& _to_n = graph.to( l );
          if ( 0 < _to_n.size() ) {
            double s = 0.0;
            for ( auto& ele : _to_n ) {
              const int &n = ele.first;
              const double &alpha = ele.second;
              for ( int j=0; j<dim; j++ ) {
                centers[l].get()[j] += feat[n][j] * alpha;
              }
              s += alpha;
            }
            algebra::scale( centers[l].get(), dim, 1.0 / s );
          }
        }

        // calculate energy
        double e = 0.0;
        {
          double D[dim];
          for ( size_t n=0; n<N; n++ ) {
            auto& _to_l = graph.from( n );
            for ( int j=0; j<dim; j++ ) {
              D[j] = feat[n][j];
            }

            
            for ( auto& ele : _to_l ) {
              const int& l = static_cast<size_t>( ele.first );
              const double& alpha = ele.second;
              algebra::minusScaledFrom( D, centers[l].get(), dim, alpha );
            }
            e += norm2( D, dim );
          }
        }
        
        Info( "Energy: %.6lf", e );

        if ( 0 < iter && fabs( e - last_e ) < options.converge ) {
          break;
        } else {
          last_e = e;
        }

        
        // update alphas
        Info( "Updating Graph ..." );
        bimap.clear();



        ProgressBar alphaProgress( N_int );
        int completed = 0;
#       pragma omp parallel for
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
          
          solveAlphas( [&dim,&N,&L,&_to_l,&n,&feat,this,&supportSize] // energy
                       ( const std::vector<double>& alphas, alphaStorage& store ) 
                       {
                         double *D = &store.D[0];
                         for ( int j=0; j<dim; j++ ) {
                           D[j] = feat[n][j];
                         }
                         for ( int i=0; i<supportSize; i++ ) {
                           size_t l = static_cast<size_t>( _to_l[i].first );
                           // D -= alpha(i) * C(l)
                           algebra::minusScaledFrom( D, centers[l].get(), dim, alphas[i] );
                         }
                         return algebra::norm2( D, dim );
                       },
                       
                       [&dim,&N,&L,&_to_l,&n,&feat,this,&supportSize] // negative derivative
                       ( std::vector<double>& d, const std::vector<double> __attribute__ ((__unused__)) &alphas, 
                         alphaStorage& store )
                       {
                         // -deriv(i) = C(l)^T D(i)
                         for ( int i=0; i<supportSize; i++ ) {
                           size_t l = static_cast<size_t>( _to_l[i].first );
                           d[i] = algebra::dotprod( centers[l].get(), &store.D[0], dim );
                         }
                       },

                       [&dim,&N,&L,&_to_l,&n,&feat,&supportSize] // projection
                       ( std::vector<double>& alphas, alphaStorage __attribute__ ((__unused__)) &store )
                       {
                         double tmp[supportSize];
                         algebra::copy( tmp, &alphas[0], supportSize );
                         algebra::watershed( tmp, &alphas[0], supportSize );
                       } );

          
          

          
          heap<double,int> ranker( options.replicate );
          
          for ( int i=0; i<supportSize; i++ ) {
            ranker.add( -solveAlphas.x[i], i );
          }

          // post projection
          double alphas[ranker.len];
          for ( int j=0; j<ranker.len; j++ ) {
            alphas[j] = solveAlphas.x[ranker[j]];
          }

          {
            double tmp[ranker.len];
            algebra::copy( tmp, alphas, ranker.len );
            algebra::watershed( tmp, alphas, ranker.len );
          }
          

          // update bimap
          for ( int j=0; j<ranker.len; j++ ) {
            if ( alphas[j] > options.epsilon ) {
              bimap.add( n, _to_l[ranker[j]].first, alphas[j] );
            }
          }

#         pragma omp critical
          {
            completed ++;
            alphaProgress.update( completed, "updating graph" );
          }
        } // end for n
        
      }
      
      n_to_l = std::move( bimap );
    }
    
  };
}
