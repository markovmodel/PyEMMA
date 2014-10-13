/* (c) 2013 Fabian Paul <fabian.paul@mpikg.mpg.de> 
 version 2013-02-22 feb */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include <locale.h>

#define D(obj,i) (((double*)((obj)->data))[i])

// TODO: move dcd reader impl to stallone (java)

/* class for file readers */
typedef struct reader_struct {
	int (*init)(struct reader_struct*, char*);
	int (*read_record)(struct reader_struct, double*, int);
	FILE *stream;
	int swap;
	unsigned charmm, charmm_4dims, charmm_extrablock, n;
	long endoffile;
} reader;

/* Split a line of numers into an array of doubles
 *  stream: input stream
 *  record: output array of doubles
 *  size:   input maximum space in record array
 *          if zero, don't fill record, just count entries
 * returns the number of doubles read into record or 0 in case
 *         of eof or -1 in case of stream read error 
 */
#define NUMBER 0
#define SPACE 1
int read_record_dat(reader r, double* record, int size) {
	int ch;
	int ocount, icount;
	int state;
	char buffer[0x100];
	char *bufptr;
	double d;

	if (ferror(r.stream))
		return -1;
	if (feof(r.stream))
		return 0;

	ocount = 0;
	icount = 0;
	ch = getc(r.stream);
	bufptr = buffer;

	if (isspace(ch))
		state = SPACE;
	else
		state = NUMBER;

	while (ch != '\n' && ch != -1) {
		switch (state) {
		case NUMBER:
			if (isdigit(ch) || ch == '.' || ch == '-' || ch == '+' || ch == 'e'
					|| ch == 'E') {
				*bufptr = ch;
				bufptr++;
				if (bufptr - buffer >= 0x100)
					return -1;
			} else if (isspace(ch) || ch == ',') {
				*bufptr = '\0';
				d = atof(buffer);
				if (size > 0 && ocount >= size)
					return -1;
				if (size > 0)
					record[ocount] = d;
				ocount++;
				icount++;
				bufptr = buffer;
				state = SPACE;
			} else
				return -1;
			break;
		case SPACE:
			if (!isspace(ch)) {
				state = NUMBER;
				ungetc(ch, r.stream);
			}
			break;
		}
		ch = getc(r.stream);
	}
	if (bufptr != buffer) { /* last number was not converted */
		*bufptr = '\0';
		d = atof(buffer);
		if (size > 0 && ocount >= size)
			return -1;
		if (size > 0)
			record[ocount] = d;
		ocount++;
	}

	if (ferror(r.stream))
		return -1;
	else
		return ocount;
}

int init_dat(reader *r, char *fname) {
	int n;
	r->stream = fopen(fname, "r");
	if (!r->stream)
		return -1;
	n = read_record_dat(*r, NULL, 0);
	if (n < 0)
		return n;
	else {
		rewind(r->stream);
		return n;
	}
}

float reverse_float(const float x) {
	float result;
	char *input = (char*) &x;
	char *output = (char*) &result;

	output[0] = input[3];
	output[1] = input[2];
	output[2] = input[1];
	output[3] = input[0];

	return result;
}

unsigned long reverse_int(const unsigned long x) {
	unsigned long result;
	char *input = (char*) &x;
	char *output = (char*) &result;

	output[0] = input[3];
	output[1] = input[2];
	output[2] = input[1];
	output[3] = input[0];

	return result;
}

int read_int(unsigned int *value, FILE *stream, int swap) {
	int i;
	if (fread(&i, 4, 1, stream) != 1)
		return -1;
	if (swap)
		*value = reverse_int(i);
	else
		*value = i;
	return 1;
}

int read_float(double *value, FILE *stream, int swap) {
	float f;
	if (fread(&f, 4, 1, stream) != 1)
		return -1;
	if (swap)
		*value = (double) reverse_float(f);
	else
		*value = (double) f;
	return 1;
}

int init_dcd(reader *r, char *fname) {
	unsigned key, nset, istart, nsavc, i, newsize, numlines, namnf;
	char cord[4];

	printf("using dcd reader ["); // debug

	r->stream = fopen(fname, "r");
	if (!r->stream)
		return -1;

	if (fread(&key, 4, 1, r->stream) != 1)
		return -1;
	if (key != 84) {
		r->swap = 1;
		printf("byte order is reversed, "); // debug
	} else
		r->swap = 0;

	// Figure out how big the file is
	if (fseek(r->stream, 0, SEEK_END) != 0)
		return -1;
	r->endoffile = ftell(r->stream);
	if (r->endoffile == -1)
		return -1;
	if (fseek(r->stream, 4, SEEK_SET) != 0)
		return -1;

	// Read CORD
	if (fread(cord, 1, 4, r->stream) != 4)
		return -1;
	printf("sentinel %c%c%c%c, ", cord[0], cord[1], cord[2], cord[3]); // debug
	// Read NSET
	if (read_int(&nset, r->stream, r->swap) != 1)
		return -1;
	if (read_int(&istart, r->stream, r->swap) != 1)
		return -1;
	if (read_int(&nsavc, r->stream, r->swap) != 1)
		return -1;

	// Reposition to 40 from beginning; read namnf, number of free atoms
	printf("test namnf, "); // debug
	if (fseek(r->stream, 40, SEEK_SET) != 0)
		return -1;
	if (read_int(&namnf, r->stream, r->swap) != 1)
		return -1;
	if (namnf != 0) {
		fprintf(stderr, "namnf file format not supported.\n");
		return -1;
	}

	// Figure out if we're CHARMm or not
	if (fseek(r->stream, 84, SEEK_SET) != 0)
		return -1;
	if (read_int(&i, r->stream, r->swap) != 1)
		return -1;
	printf("charmm? %d, ", i != 0); // debug
	if (i == 0)
		r->charmm = 0;
	else {
		r->charmm = 1;
		r->charmm_extrablock = 0;
		r->charmm_4dims = 0;
		// check for extra block
		if (fseek(r->stream, 48, SEEK_SET) != 0)
			return -1;
		if (read_int(&i, r->stream, r->swap) != 1)
			return -1;
		if (i == 1)
			r->charmm_extrablock = 1;
		printf("extrablock %d, ", r->charmm_extrablock); // debug
		// check for 4 dims
		if (fseek(r->stream, 52, SEEK_SET) != 0)
			return -1;
		if (read_int(&i, r->stream, r->swap) != 1)
			return -1;
		if (i == 1)
			r->charmm_4dims = 1;
		printf("charmm_4dims %d, ", r->charmm_4dims); // debug
	}

	// Get the size of the next block, and skip it
	// This is the title
	if (fseek(r->stream, 92, SEEK_SET) != 0)
		return -1;
	if (read_int(&newsize, r->stream, r->swap) != 1)
		return -1;
	//printf("newsize %d, ", newsize); // debug
	if (read_int(&numlines, r->stream, r->swap) != 1)
		return -1;
	//printf("numlines %d, ", numlines); // debug
	if (fseek(r->stream, numlines * 80, SEEK_CUR) != 0)
		return -1;
	if (read_int(&newsize, r->stream, r->swap) != 1)
		return -1;
	//printf("newsize %d, ", newsize); // debug

	// Read in a 4, then the number of atoms, then another 4
	if (read_int(&i, r->stream, r->swap) != 1)
		return -1;
	if (read_int(&r->n, r->stream, r->swap) != 1)
		return -1;
	if (read_int(&i, r->stream, r->swap) != 1)
		return -1;
	printf("natoms %d ]\n", r->n); // debug

	return r->n * 3;
}

int read_record_dcd(reader r, double* record, int size) {
	unsigned blocksize;
	int i;
	long n;

	n = ftell(r.stream);
	//printf("we are at %d, end is at %d\n", n, r.endoffile); // debug
	if (n > r.endoffile) {
		fprintf(stderr, "overshot file end\n");
		return -1;
	}
	if (n == -1)
		return -1;
	if (n >= r.endoffile)
		return 0;

	// If this is a CHARMm file and contains an extra data block, we must skip it
	if (r.charmm && r.charmm_extrablock) {
		//printf("skipping block\n"); // debug
		if (read_int(&blocksize, r.stream, r.swap) != 1)
			return -1;
		if (fseek(r.stream, blocksize, SEEK_CUR) != 0)
			return -1;
		if (read_int(&blocksize, r.stream, r.swap) != 1)
			return -1;
	}

	// Get x coordinates
	if (read_int(&blocksize, r.stream, r.swap) != 1)
		return -1;
	//printf("read X %d\n", blocksize); // debug
	if (3 * blocksize / 4 > size)
		return -1;
	for (i = 0; i < blocksize / 4; i++)
		if (read_float(&record[3 * i], r.stream, r.swap) != 1)
			return -1;
	if (read_int(&blocksize, r.stream, r.swap) != 1)
		return -1;
	//printf("read X %d\n", blocksize); // debug

	// Get y coordinates
	if (read_int(&blocksize, r.stream, r.swap) != 1)
		return -1;
	//printf("read Y %d\n", blocksize); // debug
	if (3 * blocksize / 4 > size)
		return -1;
	for (i = 0; i < blocksize / 4; i++)
		if (read_float(&record[3 * i + 1], r.stream, r.swap) != 1)
			return -1;
	if (read_int(&blocksize, r.stream, r.swap) != 1)
		return -1;
	//printf("read Y %d\n", blocksize); // debug

	// Get z coordinates
	if (read_int(&blocksize, r.stream, r.swap) != 1)
		return -1;
	//printf("read Z %d\n", blocksize); // debug
	if (3 * blocksize / 4 > size)
		return -1;
	for (i = 0; i < blocksize / 4; i++)
		if (read_float(&record[3 * i + 2], r.stream, r.swap) != 1)
			return -1;
	if (read_int(&blocksize, r.stream, r.swap) != 1)
		return -1;
	//printf("read Z %d\n", blocksize); // debug

	// Skip the 4th dimension, if present
	if (r.charmm && r.charmm_4dims) {
		//printf("skip 4D\n"); // debug
		if (read_int(&blocksize, r.stream, r.swap) != 1)
			return -1;
		if (fseek(r.stream, blocksize, SEEK_CUR) != 0)
			return -1;
		if (read_int(&blocksize, r.stream, r.swap) != 1)
			return -1;
	}

	return 3 * blocksize / 4;
}

reader reader_by_fname(char *fname) {
	reader r;
	int l;
	l = strlen(fname);
	if (l >= 4 && fname[l - 4] == '.' && tolower(fname[l - 3]) == 'd'
			&& tolower(fname[l - 2]) == 'c' && tolower(fname[l - 1]) == 'd') {
		r.init = init_dcd;
		r.read_record = read_record_dcd;
	} else {
		r.init = init_dat;
		r.read_record = read_record_dat;
	}
	return r;
}

/**
 * TODO: instead of usage of a dictionary parameterize a java funciton
 * with a list of filenames, calculate everthing on java side an return a
 * matrix to python.
 */
PyArrayObject* unpack_array(PyObject *obj, int dim1, int dim2) {
	PyArrayObject *cont_array;
	cont_array = NULL;

	if (!obj)
		goto error;

	// scalar
	if (dim1 == 1 && dim2 == 1) {
		cont_array = (PyArrayObject*) (PyArray_ContiguousFromAny(obj,
				PyArray_DOUBLE, 0, 0));
		if (!cont_array)
			goto error;
	} else { // vector
		if (dim2 == 1) {
			cont_array = (PyArrayObject*) (PyArray_ContiguousFromAny(obj,
					PyArray_DOUBLE, 1, 1));
			if (!cont_array)
				goto error;
			if (cont_array->dimensions[0] != dim1) {
				PyErr_SetString(PyExc_ValueError,
						"Number of order parameters and length of mean/std/weights differ. Forgot time column?");
				goto error;
			}
		} else { // matrix
			cont_array = (PyArrayObject*) (PyArray_ContiguousFromAny(obj,
					PyArray_DOUBLE, 2, 2));
			if (!cont_array)
				goto error;
			if (cont_array->dimensions[0] != dim1
					|| cont_array->dimensions[1] != dim2) {
				PyErr_SetString(PyExc_ValueError,
						"Number of order parameters and length of mean/std/weights differ. Forgot time column?");
				goto error;
			}
		}
	}
	/* fall through */

	error: return cont_array;
}

PyArrayObject* new_array(int dim1, int dim2) {
	PyObject *array;
	PyArrayObject *cont_array;
	npy_intp dims[2];
	int i;

	array = NULL;
	cont_array = NULL;
	dims[0] = dim1;
	dims[1] = dim2;
	if (dim1 == 1 && dim2 == 1)
		array = PyArray_SimpleNew(0, dims, PyArray_DOUBLE);
	else if (dim2 == 1)
		array = PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
	else
		array = PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
	if (!array)
		goto error;
	cont_array = (PyArrayObject*) PyArray_ContiguousFromAny(array,
			PyArray_DOUBLE, 0, 2);
	if (!cont_array)
		goto error;
	Py_DECREF(array);
	/* initialize array content to zero */
	for (i = 0; i < dim1 * dim2; i++)
		D(cont_array,i) = 0.0;
	return cont_array;

	error: if (array)
		Py_DECREF(array);
	if (cont_array)
		Py_DECREF(cont_array);
	return NULL;
}

PyArrayObject* unpack_or_make(PyObject *dict, char *name, int dim1, int dim2) {
	PyObject *array;
	PyArrayObject *cont_array;

	cont_array = NULL;
	array = NULL;

	array = PyDict_GetItemString(dict, name); /*b*/
	if (!array) { /* make default object */
		cont_array = new_array(dim1, dim2);
		if (!cont_array)
			goto error;
		/* store a ref to the new data in the dict */
		if (PyDict_SetItemString(dict, name, (PyObject*) cont_array) != 0)
			goto error;
	} else { /* extract data from dict */
		cont_array = unpack_array(array, dim1, dim2);
		if (!cont_array)
			goto error;
		/* If unpack_array happened to copy the array to make it contiguous
		 we need to store it again in the dict. We make our lives easier
		 by always storing it again, regardless of what happened. */
		if (PyDict_SetItemString(dict, name, (PyObject*) cont_array) != 0)
			goto error;
	}

	return cont_array;

	error: if (cont_array)
		Py_DECREF(cont_array);
	return NULL;
}

void update_mean(const double *record, PyArrayObject* mean, int n) {
	int i;
	#pragma omp parallel for private(i)
	for (i = 0; i < n; i++)
		D(mean,i) += record[i];
}

void update_var(const double *record, PyArrayObject* var, int n) {
	int i;
	#pragma omp parallel for private(i)
	for (i = 0; i < n; i++)
		D(var,i) += record[i] * record[i];
}

void update_cov(const double *record, PyArrayObject* cov, int n) {
	int i, j;
	#pragma omp parallel for private(i,j)
	for (i = 0; i < n; i++) {
		for (j = i; j < n; j++) {
			D(cov,i*n+j) += record[i] * record[j];
		}
	}
}

void update_tcov(const double *record, PyArrayObject* tcov,
		PyArrayObject* samples, int n, int m, double *shift, int lag) {
	int i, j;

	if (m >= lag) {
		D(samples,0) += 1;
		#pragma omp parallel for private(i,j)
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				D(tcov,i*n+j) += record[j]
						* shift[((m - lag) % lag) * n + i];
			}
		}
	}
	#pragma omp parallel for private(i)
	for (i = 0; i < n; i++)
		shift[(m % lag) * n + i] = record[i];
}

/**
 * calculates covariance matrices
 */
static PyObject *run(PyObject *self, PyObject *args) {
	char *fname;
	int do_mean, do_var, do_cov, do_tcov;
	int lag;
	int i, j, n, c;
	PyObject *stats;
	PyArrayObject* mean, *var, *cov, *tcov, *samples, *tcov_samples;
	double *record, *shift;
	reader input;

	record = shift = NULL;
	mean = var = cov = tcov = samples = tcov_samples = NULL;
	input.stream = NULL;
	stats = NULL;

	if (!PyArg_ParseTuple(args, "sO!iiiii", &fname, &PyDict_Type, &stats,
			&do_mean, &do_var, &do_cov, &do_tcov, &lag))
		goto error;

	setlocale(LC_NUMERIC, "C");
	/* open date file and get record size */
	input = reader_by_fname(fname);
	n = input.init(&input, fname);
	if (n < 1) {
		PyErr_SetFromErrno(PyExc_IOError);
		goto error;
	}

	/* allocate memory for a row */
	record = (double*) malloc(n * sizeof(double));
	if (!record) {
		PyErr_NoMemory();
		goto error;
	}

	/* setup data structures */
	mean = unpack_or_make(stats, "mean", n, 1);
	if (!mean)
		goto error;
	if (do_var) {
		var = unpack_or_make(stats, "var", n, 1);
		if (!var)
			goto error;
	}
	if (do_cov) {
		cov = unpack_or_make(stats, "cov", n, n);
		if (!cov)
			goto error;
	}
	if (do_tcov) {
		if (lag < 1) {
			PyErr_SetString(PyExc_ValueError, "lag < 1 not allowed.");
			goto error;
		}
		tcov = unpack_or_make(stats, "tcov", n, n);
		if (!tcov)
			goto error;
		tcov_samples = unpack_or_make(stats, "tcov_samples", 1, 1);
		if (!tcov_samples)
			goto error;
		shift = (double*) malloc(n * lag * sizeof(double));
		if (!shift) {
			PyErr_NoMemory();
			goto error;
		}
		for (i = 0; i < n * lag; i++)
			shift[i] = 0;
	}

	samples = unpack_or_make(stats, "samples", 1, 1);
	if (!samples)
		goto error;

	/* iterate over file and calculate everything */
	j = 0;
	while (1) {
		if (ferror(input.stream)) {
			PyErr_SetFromErrno(PyExc_IOError);
			goto error;
		}
		c = input.read_record(input, record, n);
		if (c == 0)
			break; /* empty line or eof */
		if (c != n) {
			PyErr_SetString(PyExc_ValueError,
					"Unexpected line format or read error.");
			goto error;
		}

		/* debug */
		//for(i=0;i<n;i++) printf("%f, ",record[i]);
		//printf("\n");
		if (!do_mean)
			for (i = 0; i < n; i++)
				record[i] -= D(mean,i);

		if (do_mean)
			update_mean(record, mean, n);
		if (do_var)
			update_var(record, var, n);
		if (do_cov)
			update_cov(record, cov, n);
		if (do_tcov)
			update_tcov(record, tcov, tcov_samples, n, j, shift, lag);

		j++;
	}

	D(samples,0) += j;

	/* fill lower half of covariance matrix */
	if (do_cov) {
		#pragma omp parallel for private(i,j)
		for (i = 0; i < n; i++)
			for (j = i + 1; j < n; j++)
				D(cov,j*n+i) = D(cov,i*n+j);
	}

	if (input.stream)
		fclose(input.stream);
	if (record)
		free(record);
	if (shift)
		free(shift);
	if (samples)
		Py_DECREF(samples);
	if (mean)
		Py_DECREF(mean);
	if (var)
		Py_DECREF(var);
	if (cov)
		Py_DECREF(cov);
	if (tcov)
		Py_DECREF(tcov);
	if (tcov_samples)
		Py_DECREF(tcov_samples);

	return Py_BuildValue(""); /* =None */

	error: if (input.stream)
		fclose(input.stream);
	if (record)
		free(record);
	if (shift)
		free(shift);
	if (samples)
		Py_DECREF(samples);
	if (mean)
		Py_DECREF(mean);
	if (var)
		Py_DECREF(var);
	if (cov)
		Py_DECREF(cov);
	if (tcov)
		Py_DECREF(tcov);
	if (tcov_samples)
		Py_DECREF(tcov_samples);

	return NULL;
}

static PyObject *project(PyObject *self, PyObject *args) {
	FILE *out;
	char *fin, *fout;
	PyObject *res;
	double *record;
	PyArrayObject *mean, *W;
	PyObject *py_mean, *py_W;
	int i, j, n, m, c;
	int max, time_column;
	double sum;
	reader input;

	input.stream = out = NULL;
	res = NULL;
	mean = W = NULL;
	record = NULL;

	if (!PyArg_ParseTuple(args, "ssOOii", &fin, &fout, &py_mean, &py_W, &max,
			&time_column))
		goto error;

	time_column = !!time_column; /* convert to boolean */
	if (max < 1) {
		PyErr_SetString(PyExc_ValueError, "max < 1");
		goto error;
	}

	// set locale to c, to get proper formatted floating point numbers.
	setlocale(LC_NUMERIC, "C");

	/* open input file and get record size */
	input = reader_by_fname(fin);
	n = input.init(&input, fin);
	if (n < 1) {
		PyErr_SetFromErrno(PyExc_IOError);
		goto error;
	}

	m = n - time_column; /* length of actual data */

	if (max > m) {
		PyErr_SetString(PyExc_ValueError,
				"requested output record length (max) > input record length");
		goto error;
	}

	/* convert numeric parameters to arrays */
	mean = unpack_array(py_mean, m, 1);
	W = unpack_array(py_W, m, m);
	if (!mean || !W)
		goto error;

	/* allocate memory */
	record = (double*) malloc(n * sizeof(double));
	if (!record) {
		PyErr_NoMemory();
		goto error;
	}

	/* open output file */
	out = fopen(fout, "w");
	if (!out) {
		PyErr_SetFromErrno(PyExc_IOError);
		goto error;
	}	

	/* iterate over input */
	while (1) {
		if (ferror(input.stream)) {
			PyErr_SetFromErrno(PyExc_IOError);
			goto error;
		}
		c = input.read_record(input, record, n);
		if (c == 0)
			break; /* empty line or eof */
		if (c != n) {
			PyErr_SetString(PyExc_ValueError,
					"Unexpected line format or read error.");
			goto error;
		}

		#pragma omp parallel for private(i)
		for (i = 0; i < m; i++)
			record[i] = record[i + time_column] - D(mean,i);
		for (j = 0; j < max; j++) {
			sum = 0;
			#pragma omp parallel for reduction(+:sum)
			for (i = 0; i < m; i++)
				sum += record[i] * D(W,m*i+j);
			fprintf(out, "%f", sum);
			if (j != max - 1)
				fprintf(out, " ");
		}
		fprintf(out, "\n");
	}

	res = Py_BuildValue(""); /* =None */

	error: if (input.stream)
		fclose(input.stream);
	if (out)
		fclose(out);
	if (record)
		free(record);
	if (mean)
		Py_DECREF(mean);
	if (W)
		Py_DECREF(W);
	return res;
}

#define RUN_USAGE \
  "Calculate mean, variance, correlation and time-lagged correlation.\n" \
  "run(fname, d, mean, var, cov, tcov, lag)\n\n" \
  "fname: name of file that contains tabulated data\n" \
  "d:     a dictionary that holds the statistical quantities.\n" \
  "       Can be empty at first or can contain partial results from parts\n" \
  "       of the trajectory computed in a previous run.\n" \
  "mean: (bool) if true, calculate mean\n" \
  "             if false, subtract dictionary entry \'mean\' from every\n" \
  "             frame before calculation of other statistical quantities\n" \
  "             Simultaneous calculation of mean and other quantities of\n" \
  "             uncentered data is *not* possible.\n" \
  "var:  (bool) calculate varaince?\n" \
  "cov:  (bool) calculate covariance matrix?\n" \
  "tcov: (bool) calculate time-lagged covariance matrix?\n" \
  "The statistical quantities can then be accessed using the dictionary\n" \
  "keys \'mean\', \'var\', \'cov\', \'tcov\', \'samples\' and\n" \
  " \'tcov_samples\'.\n" \
  "Note that statistical estimates are not normalized. To calculate \n" \
  "the actual mean, you need to compute d[\'mean\']/d[\'samples\'] and so on."

#define PROJECT_USAGE \
  "Open file with tabulated data. Center and project each line.\n" \
  "project(fin, fout, mean, weights, max_proj, time_column)\n\n" \
  "fin:  name of input file\n" \
  "fout: name of output file (for projected data)\n" \
  "mean: list of means; is subtracted from every line\n" \
  "weights: n x n square matrix of weights W; the ith entry of each new\n" \
  "         line is y_i = Sum_k W_ki x_k, where x_k are the entries of\n" \
  "         the input line after centering.\n" \
  "max_proj: only write y_0 through y_{max_proj-1} to the output file\n" \
  "time_colum: (bool) input data contains time column?" \

static PyMethodDef cocoMethods[] =
		{ { "run", run, METH_VARARGS, RUN_USAGE }, { "project", project,
				METH_VARARGS, PROJECT_USAGE }, { NULL, NULL, 0, NULL } };

PyMODINIT_FUNC initcocovar(void) {
	(void) Py_InitModule("cocovar", cocoMethods);
	import_array();


}
