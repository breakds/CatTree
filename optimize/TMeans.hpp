#pragma once

#include <vector>
#include "LLPack/algorithms/heap.hpp"
#include "../data/vector.hpp"
#include "../data/Bipartite.hpp"


namespace cat_tree {
  namespace TMeans {
    
    namespace {


      template <typename feature_t, typename dataType = float>
      void CenterMeans( std::vector<dataType> &centers, 
                        const std::vector<feature_t> &feat,
                        const std::vector<int> &ind,
                        int dim,
                        Bipartite& n_to_l )
      {
      
        int N = n_to_l.sizeA();
        int L = n_to_l.sizeB();
        
        std::fill( centers.begin(), centers.end(), 0.0 );
        
        std::vector<int> count( L, 0 );
        
        for ( int n=0; n<N; n++ ) {
          auto _to_l = n_to_l.getToSet( n );
          for ( auto& ele : _to_l ) {
            int l = ele.first;
            count[l]++;
            for ( int j=0; j<dim; j++ ) {
              centers[l*dim+j] += feat[ind[n]][j];
            }
          }
        }
        
        for ( int l=0; l<L; l++ ) {
          if ( 0 < count[l] ) {
            scale( &centers[l*dim], dim, static_cast<float>( 1.0 / count[l] ) );
          }
        }
      }
    }

    struct Options
    {
      int maxIter;
      int replicate;
      double converge;
      Options() : maxIter(100), replicate(10), converge(1e-5) {}
    };

    template<typename feature_t, typename dataType = float>
    void Clustering( const std::vector<feature_t> &feat,
                     const std::vector<int> &ind,
                     int dim,
                     Bipartite& n_to_l,
                     Options &options )
    {
      int N = n_to_l.sizeA();
      int L = n_to_l.sizeB();

      std::vector<dataType> centers( L * dim, 0.0 );

      CenterMeans( centers, feat, ind, dim, n_to_l );

      Bipartite bimap( N, L );

      double lastEnergy = 0.0;
              
      for ( int iter=0; iter<options.maxIter; iter++ ) {
        
        bimap.clear();
        
        Info( "TMeans iter %d", iter );
        // pick centers
        for ( int n=0; n<N; n++ ) {
          auto _to_l = n_to_l.getToSet( n );
          heap<double,int> ranker( options.replicate );
          for ( auto& ele : _to_l ) {
            int l = ele.first;
            double dist = 0.0;
            for ( int j=0; j<dim; j++ ) {
              double tmp = centers[l*dim+j] - feat[ind[n]][j];
              dist += tmp * tmp;
            }
            ranker.add( dist, l );
          }

          for ( int j=0; j<ranker.len; j++ ) {
            bimap.add( n, ranker[j], 1.0 / options.replicate );
          }
          
        } // end for n

        CenterMeans( centers, feat, ind, dim, bimap );

        // Calculate Energy
        double energy = 0.0;
        for ( int n=0; n<N; n++ ) {
          auto _to_l = bimap.getToSet( n );
          for ( auto& ele : _to_l ) {
            int l = ele.first;
            for ( int j=0; j<dim; j++ ) {
              double tmp = centers[l*dim+j] - feat[ind[n]][j];
              energy += tmp * tmp;
            }
          }
        }
        
        if ( 0 <iter && fabs(lastEnergy-energy) < options.converge ) {
          break;
        }
        
        lastEnergy = energy;
        
        printf( "Energy: %.5lf\n", energy );
        
      } // end for iter
      
      n_to_l = std::move( bimap );
    }
  }
}
