#pragma once

#include <cassert>
#include <vector>
#include <string>
#include <unordered_map>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/LispFormParser.hpp"

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
      className.clear();
      colors.clear();
      int id = 0;
      for ( auto& every : lisp ) {
        _to_name.push_back( every );
        _to_id.insert( std::make_pair( every, id ) );
        wt[id] = 1.0 / lisp[every].toInt();
        id ++;
      }
      classes = static_cast<int>( className.size() );
      inv = 1.0 / classes;
    }

    void Summary()
    {
      if ( -1 == classes ) {
        Error( "LabelSet has not been initialized yet." );
        exit( -1 );
      }
      Info( "Label Set Summary ..." );
      for ( int i=0; i<classes; i++ ) {
        printf( "%3d)%20s\t\t%5lf\n"
                i,
                _to_name[i].c_str(),
                wt[i] );
      }
    }
  };
}
