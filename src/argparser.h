#include <bits/stdc++.h>
#include <iostream>

typedef std::unordered_map<std::string, std::string> umss;
typedef std::unordered_map<std::string, int> umsi;
typedef std::vector<std::string> vs;

class ArgumentParser {
  umss args;
  vs arg_list;
  umsi arg_names;
  std::string FUNCTION_NAME = "function_name";

public:
  ArgumentParser();
  void parse_args(int argc, char *argv[]);
  void add_argument(const std::string &argname);
  std::string get_argument(const std::string &argname);
};