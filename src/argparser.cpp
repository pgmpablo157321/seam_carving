#include "argparser.h"

void ArgumentParser::add_argument(const std::string &argname) {

  assert((argname != FUNCTION_NAME) &&
         "Argument name is reserved for special arguments");
  assert((arg_names.find(argname) == arg_names.end()) &&
         "Argument name already exists");
  arg_names[argname] = 1;
  arg_list.push_back(argname);
}

ArgumentParser::ArgumentParser() {}

void ArgumentParser::parse_args(int argc, char *argv[]) {
  for (int i = 0; i < argc; i++) {
    if (i == 0) {
      args[FUNCTION_NAME] = argv[i];
    } else if (i % 2 == 1) {
      assert(arg_names.find(argv[i]) != arg_names.end() &&
             "Argument name not found");
    } else {
      args[argv[i - 1]] = argv[i];
    }
  }
  for (int i = 0; i < arg_list.size(); i++) {
    assert(args.find(arg_list[i]) != args.end() &&
           "A required argument was not provided");
  }
}

std::string ArgumentParser::get_argument(const std::string &argname) {
  assert((args.find(argname) != args.end()) && "Argument does not exists");
  return args[argname];
}