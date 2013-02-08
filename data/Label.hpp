#pragma once

#include <cassert>
#include <vector>
#include <string>
#include <unordered_map>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/LispFormParser.hpp"
#include "vector.hpp"

using std::vector;
using std::string;
using std::unordered_map;



namespace cat_tree
{

  namespace LabelSet
  {
    int classes = -1; // uninitialized
    double inv = 1.0;
    namespace {
      // properties of LabelSet
      vector<string> _to_name;
      unordered_map<string,int> _to_id;
      vector<double> wt;
    };

    void initialize( string filename )
    {
      LispFormParser lisp;
      lisp.parse( filename );
      _to_name.clear();
      _to_id.clear();
      int id = 0;
      for ( auto& every : lisp ) {
        _to_name.push_back( every );
        _to_id.insert( std::make_pair( every, id ) );
        wt.push_back( 1.0 / lisp[every].toInt() );
        id ++;
      }

      classes = static_cast<int>( _to_name.size() );
      inv = 1.0 / classes;

      // normalize the weights
      double s = 1.0 / sum_vec( &wt[0], classes );
      for ( auto& w : wt ) {
        w *= s;
      }
    }

    void Summary()
    {
      if ( -1 == classes ) {
        Error( "LabelSet has not been initialized yet." );
        exit( -1 );
      }
      Info( "Label Set Summary ..." );
      for ( int i=0; i<classes; i++ ) {
        printf( "%3d)%20s\t\t%5lf\n",
                i,
                _to_name[i].c_str(),
                wt[i] );
      }
    }

    int GetClass( const string& str ) {
      // search for termination character
      int slashpos = str.find_first_of( '/' );
      if ( -1 == slashpos ) {
        return _to_id[str];
      }
      return _to_id[str.substr( 0, slashpos )];
    }
  };
}
