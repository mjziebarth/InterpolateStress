project('InterpolateStress','cpp','cython',
        default_options : ['optimization=3'])
add_project_arguments(['-march=native','-mtune=native'], language : 'cpp')

geographiclib_dep = dependency('geographiclib')

include = include_directories('cpp/include')

libinterpolatestress = static_library(
               'interpolatestress',
               ['cpp/src/kernel.cpp', 'cpp/src/data.cpp', 'cpp/src/api.cpp'],
               dependencies : [geographiclib_dep, dependency('openmp')],
               include_directories : include)

python = import('python').find_installation()
dep_py = python.dependency()

python.extension_module('backend',
                        'interpolatestress/backend.pyx',
                        dependencies : dep_py,
                        include_directories : include,
                        override_options : ['cython_language=cpp'],
                        link_with : libinterpolatestress)