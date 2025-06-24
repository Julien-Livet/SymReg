QT += testlib

CONFIG += c++23 object_parallel_to_source

QMAKE_CXXFLAGS += -std=c++23
QMAKE_LFLAGS += -Wl,-rpath,/usr/local/lib

CONFIG(debug, debug|release) {
} else {
    DEFINES += NDEBUG
}

SOURCES = testsymreg.cpp

HEADERS = BinaryOperator.h \
    Expression.h \
    UnaryOperator.h \
    SymbolicRegressor.h \
    Variable.h

LIBS += -lceres -lcholmod -lblas -llapack -lglog -labsl_base -labsl_city -labsl_civil_time -labsl_cord -labsl_cord_internal -labsl_cordz_functions -labsl_cordz_handle -labsl_cordz_info -labsl_cordz_sample_token -labsl_crc32c -labsl_crc_cord_state -labsl_crc_cpu_detect -labsl_crc_internal -labsl_debugging_internal -labsl_decode_rust_punycode -labsl_demangle_internal -labsl_demangle_rust -labsl_die_if_null -labsl_examine_stack -labsl_exponential_biased -labsl_failure_signal_handler -labsl_flags_commandlineflag -labsl_flags_commandlineflag_internal -labsl_flags_config -labsl_flags_internal -labsl_flags_marshalling -labsl_flags_parse -labsl_flags_private_handle_accessor -labsl_flags_program_name -labsl_flags_reflection -labsl_flags_usage -labsl_flags_usage_internal -labsl_graphcycles_internal -labsl_hash -labsl_hashtablez_sampler -labsl_int128 -labsl_kernel_timeout_internal -labsl_leak_check -labsl_log_flags -labsl_log_globals -labsl_log_initialize -labsl_log_internal_check_op -labsl_log_internal_conditions -labsl_log_internal_fnmatch -labsl_log_internal_format -labsl_log_internal_globals -labsl_log_internal_log_sink_set -labsl_log_internal_message -labsl_log_internal_nullguard -labsl_log_internal_proto -labsl_log_internal_structured_proto -labsl_log_severity -labsl_log_sink -labsl_malloc_internal -labsl_periodic_sampler -labsl_poison -labsl_random_distributions -labsl_random_internal_distribution_test_util -labsl_random_internal_entropy_pool -labsl_random_internal_platform -labsl_random_internal_randen -labsl_random_internal_randen_hwaes -labsl_random_internal_randen_hwaes_impl -labsl_random_internal_randen_slow -labsl_random_internal_seed_material -labsl_random_seed_gen_exception -labsl_random_seed_sequences -labsl_raw_hash_set -labsl_raw_logging_internal -labsl_scoped_set_env -labsl_spinlock_wait -labsl_stacktrace -labsl_status -labsl_statusor -labsl_strerror -labsl_str_format_internal -labsl_strings -labsl_strings_internal -labsl_string_view -labsl_symbolize -labsl_synchronization -labsl_throw_delegate -labsl_time -labsl_time_zone -labsl_tracing_internal -labsl_utf8_for_code_point -labsl_vlog_config_internal
