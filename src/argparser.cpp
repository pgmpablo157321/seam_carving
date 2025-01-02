#include "argparser.h"

ArgumentParser::ArgumentParser() {}

void ArgumentParser::add_argument(const std::string &argname,
                                  const std::string &default_value,
                                  bool required) {

  assert((argname != FUNCTION_NAME) &&
         "Argument name is reserved for special arguments");
  assert((arg_names.find(argname) == arg_names.end()) &&
         "Argument name already exists");
  arg_names[argname] = default_value;
  arg_list.push_back(argname);
  if (required) {
    required_args.push_back(argname);
  }
}

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
    if (args.find(arg_list[i]) != args.end() && arg_names[arg_list[i]] != "") {
      args[arg_list[i]] = arg_names[arg_list[i]];
    }
  }
  for (int i = 0; i < required_args.size(); i++) {
    assert(args.find(required_args[i]) != args.end() &&
           "A required argument was not provided");
  }
}

std::string ArgumentParser::get_argument(const std::string &argname) {
  if (argname == "--config" && args.find(argname) == args.end()) {
    return "";
  }
  assert((args.find(argname) != args.end()) && "Argument does not exists");
  return args[argname];
}