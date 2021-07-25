# @file Makefile
# @author David Seka 

CC = gcc
src = ./src/
bin = ./bin/
flags = -std=c99 -pedantic -Wall -D_DEFAULT_SOURCE -D_BSD_SOURCE -D_SVID_SOURCE -D_POSIX_C_SOURCE=200809L -O3
link_openblas = -L/opt/openblas/lib -lm -lpthread -lgfortran -lopenblas

.PHONY: all clean
all: number_guesser number_guesser_unsup import_and_print_guess import_and_print_guess_unsup unsupervised_trainer apply_unsup unsupervised_tester ng_unsup_unary ipg_unsup_unary

# @: the target, ^all dependencies, < first dependency
number_guesser: $(bin)number_guesser.o $(bin)network.o $(bin)network_trainer.o -lm
	$(CC) -o $@ $^ $(link_openblas)

number_guesser_unsup: $(bin)number_guesser_unsup.o $(bin)network.o $(bin)network_trainer.o -lm
	$(CC) -o $@ $^ $(link_openblas)

import_and_print_guess: $(bin)import_and_print_guess.o $(bin)network.o $(bin)network_trainer.o -lm
	$(CC) -o $@ $^ $(link_openblas)

import_and_print_guess_unsup: $(bin)import_and_print_guess_unsup.o $(bin)network.o $(bin)network_trainer.o -lm
	$(CC) -o $@ $^ $(link_openblas)

unsupervised_trainer: $(bin)unsupervised_trainer.o $(bin)network.o $(bin)network_trainer.o -lm
	$(CC) -o $@ $^ $(link_openblas)

apply_unsup: $(bin)apply_unsup.o $(bin)network.o $(bin)network_trainer.o -lm
	$(CC) -o $@ $^ $(link_openblas)	

unsupervised_tester: $(bin)unsupervised_tester.o $(bin)network.o $(bin)network_trainer.o -lm
	$(CC) -o $@ $^ $(link_openblas)

ng_unsup_unary: $(bin)ng_unsup_unary.o $(bin)network.o $(bin)network_trainer.o -lm
	$(CC) -o $@ $^ $(link_openblas)

ipg_unsup_unary: $(bin)ipg_unsup_unary.o $(bin)network.o $(bin)network_trainer.o -lm
	$(CC) -o $@ $^ $(link_openblas)

$(bin)number_guesser.o: $(src)number_guesser.c $(src)network.c
	$(CC) $(flags) -c -o $@ $<

$(bin)network.o: $(src)network.c
	$(CC) $(flags) -c -o $@ $<

$(bin)%.o: $(src)%.c
	$(CC) $(flags) -c -o $@ $<

clean:
	rm -rf $(bin)*.o 