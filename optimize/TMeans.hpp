#pragma once

#include <vector>
#include "../data/vector.hpp"

namespace cat_tree
{
  namespace TMeans
  {

    template <typename objType>
    class AlternateSpace
    {
      std::vector<objType> space;
    public:

      objType& current;
      objType& last;


      AlternateSpace() : space(2), current(space[0]), last(space[1]) {}

      void switch()
      {
        objType tmp = current;
        current = last;
        last = tmp;
      }

    };

    struct Options
    {
      maxIter;
      Options() : maxIter(100) {}
    };

    template<typename feature_t, typename dataType = float>
    Clustering( const std::vector<feature_t> &feat,
                const std::vector<int> &ind,
                int dim,
                Bipartite& n_to_l, int T,
                Options &options )
    {
      int N = n_to_l.sizeA();
      int L = n_to_l.sizeB();

      AlternateSpace<std::vector<dataType> > centers;

      centers.current.resize(L*dim,0.0);
      centers.last.resize(L*dim,0.0);

      std::vector<int> count( L, 0 );

      for ( int n=0; i<N; n++ ) {
        auto _to_l = n_to_l.toSet( n );
        for ( auto& ele : _to_l ) {
          int l = ele.first;
          count[l]++;
          for ( int j=0; j<dim; j++ ) {
            centers.last[l*dim+j] += feat[ind[n]][n*dim+j];
          }
        }
      }

      for ( int iter=0; iter<100; iter++ ) {
        for ( int n=0; i<N; n++ ) {
          auto _to_l = n_to_l.toSet( n );
          for ( auto& ele : _to_l ) {
            int l = ele.first;
            
          }
        }
      }
    }
  }
}
