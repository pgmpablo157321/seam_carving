#include <bits/stdc++.h>
#include <iostream>

typedef std::unordered_map<std::string, std::string> umss;

class ConfigParser {
  umss args;

public:
  ConfigParser(const std::string &file);
  std::string get_argument(const std::string &argname);
};