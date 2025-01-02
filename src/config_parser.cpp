#include "config_parser.h"
#include <fstream>

ConfigParser::ConfigParser(const std::string &file) {
  // Read from the text file
  std::ifstream f(file);
  std::string line;
  std::string delimiter = ":";
  while (getline(f, line)) {
    line.erase(std::remove(line.begin(), line.end(), '-'), line.end());
    line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
    line.erase(std::remove(line.begin(), line.end(), '"'), line.end());
    std::string argname = line.substr(0, line.find(delimiter));
    std::string argval =
        line.substr(line.find(delimiter) + delimiter.length(), line.length());
    args[argname] = argval;
  }
  f.close();
};

std::string ConfigParser::get_argument(const std::string &argname) {
  assert((args.find(argname) != args.end()) && "Argument does not exists");
  return args[argname];
};