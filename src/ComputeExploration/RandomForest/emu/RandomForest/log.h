//  Copyright (c) 2018 EMU Solutions
// ========================================================================
// File Name: log.h
// Author:    Geraud Krawezik <gkrawezik@emutechnology.com>
// ------------------------------------------------------------------------
// DESCRIPTION
//      Helpers macros to provide pretty formatting for log and debugging
// ========================================================================

#define DEBUG 1

#if DEBUG
#	define LOG_DEBUG(FORMAT, ...) \
	fprintf(stdout, "\e[1;30m[DEBUG]\e[0m [%4d] " FORMAT, __LINE__, ##__VA_ARGS__); \
	fflush(stdout);
#else
#	define LOG_DEBUG(...) ;
#endif

#define LOG_INFO(...) \
	fprintf(stdout, "\e[1;34m[INFO]\e[0m " __VA_ARGS__); \
	fflush(stdout);

#define LOG_WARNING(...) \
	fprintf(stderr, "\e[1;33m[WARNING]\e[0m " __VA_ARGS__); \
	fflush(stderr);

#define LOG_CRITICAL(...) \
	fprintf(stderr, "\e[1;31m[CRITICAL]\e[0m " __VA_ARGS__); \
	fflush(stderr);
