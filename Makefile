ARCH_BITS=64 # разрядность ОС 

EXECUTABLE=program
CC=clang
CFLAGS=-m$(ARCH_BITS)
LDFLAGS=-lm

all: program.o configure.o parsing.o learn_dataset.o core.o
	$(CC) $(CFLAGS) program.o configure.o parsing.o learn_dataset.o core.o $(LDFLAGS) -o $(EXECUTABLE)

core.o: NN_core.c
	$(CC) -c $(CFLAGS) NN_core.c -o core.o
learn_dataset.o: NN_learn_dataset.c
	$(CC) -c $(CFLAGS) NN_learn_dataset.c -o learn_dataset.o
parsing.o: NN_parsing.c
	$(CC) -c $(CFLAGS) NN_parsing.c -o parsing.o
configure.o: NN_configure.c
	$(CC) -c $(CFLAGS) NN_configure.c -o configure.o
program.o: main.c
	$(CC) -c $(CFLAGS) main.c -o program.o

clean:
	rm -rf *.o $(EXECUTABLE)