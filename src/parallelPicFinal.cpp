//============================================================================
// Name        : parallelPicFinal.cpp
// Author      : Yensy Helena Gomez Villegas, John Haiber Osorio, Jose Jaramillo
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <CL/cl.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <math.h>
using namespace std;

char* readSource(const char *sourceFilename) {
	FILE *fp;
	int err;
	int size;

	char *source;

	fp = fopen(sourceFilename, "rb");
	if (fp == NULL) {
		printf("Could not open kernel file: %s\n", sourceFilename);
		exit(-1);
	}

	err = fseek(fp, 0, SEEK_END);
	if (err != 0) {
		printf("Error seeking to end of file\n");
		exit(-1);
	}

	size = ftell(fp);
	if (size < 0) {
		printf("Error getting file position\n");
		exit(-1);
	}

	err = fseek(fp, 0, SEEK_SET);
	if (err != 0) {
		printf("Error seeking to start of file\n");
		exit(-1);
	}

	source = (char*) malloc(size + 1);
	if (source == NULL) {
		printf("Error allocating %d bytes for the program source\n", size + 1);
		exit(-1);
	}

	err = fread(source, 1, size, fp);
	if (err != size) {
		printf("only read %d bytes\n", err);
		exit(0);
	}

	source[size] = '\0';

	return source;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

int crearArchivoHistoricoParticulas(float *x, float * velocidadParticulas,
		float *fuerzaParticulas, int numeroParticulas, char *nombreArchivo) {
	FILE *pFile;
	pFile = fopen(nombreArchivo, "w");
	fprintf(pFile, "%s,%s,%s\n", "posicion", "velocidad", "fuerza");
	for (int p = 0; p < numeroParticulas; p++) {
		fprintf(pFile, "%.30f,%.30f,%.30f\n", x[p], velocidadParticulas[p],
				fuerzaParticulas[p]);
	}

	fclose(pFile);
	return EXIT_SUCCESS;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////

//Inicializa los valores del vector posición.
int aleatorioX(int numeroParticulas, float tamanoCelda, float *xIon,
		float *xElectron) {
	int i;
	for (i = 0; i < numeroParticulas; i++) {
		xElectron[i] = tamanoCelda / 2;
		xIon[i] = (i + 0.5) / numeroParticulas;

	}
	return 0;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
int aleatorioVelocidad(int numeroParticulas, float *velocidadParticulasIones,
		float *velocidadParticulasElectrones) {
	int p;
	float *auxAleatorio;

	auxAleatorio = (float *) malloc((numeroParticulas) * sizeof(float));
	if (auxAleatorio == NULL) {
		printf("No se pudo reservar memoria");
		return -1;
	}
	for (p = 0; p < numeroParticulas; p++) {
		auxAleatorio[p] = (p + 0.5) / numeroParticulas;
		velocidadParticulasIones[p] = sqrt((-2 * log(1 - auxAleatorio[p])));
		velocidadParticulasElectrones[p]
				= sqrt((-2 * log(1 - auxAleatorio[p])));
	}
	free(auxAleatorio);
	return 0;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
int prepararOpenCL(float tamanoCelda, int numeroMallas, int numeroParticulas,
		float *xIon, float *xElectron, float *numeroIonesCelda,
		float *numeroElectronesCelda, int numeroIteraciones, float cargaIon,
		float *rho, float permitividadElectrica, float *rhoPoisson,
		float *diagonalInferior, float *diagonalSuperior,
		float *diagonalPrincipal, float *phi, float *campoElectrico,
		float *campoElectricoIon, float *fuerzaParticulaIon, float xita,
		float *campoElectricoElectron, float *fuerzaParticulaElectron,
		int *vectorContadorIon, int *vectorContadorElectron,
		float *velocidadParticulaIon, float *velocidadParticulaElectron,
		float pasoTiempo) {

	clock_t start, end;
	double cpu_time_used;
	cl_int status; // use as return value for most OpenCL functions
	cl_uint numPlatforms = 0;
	cl_platform_id *platforms;

	// Query for the number of recognized platforms
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS) {
		printf("clGetPlatformIDs failed\n");
		exit(-1);
	}

	// Make sure some platforms were found
	if (numPlatforms == 0) {
		printf("No platforms detected.\n");
		exit(-1);
	}

	// Allocate enough space for each platform
	platforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));
	if (platforms == NULL) {
		perror("malloc");
		exit(-1);
	}

	// Fill in platforms
	clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS) {
		printf("clGetPlatformIDs failed\n");
		exit(-1);
	}

	// Print out some basic information about each platform

	printf("%u platforms detected\n", numPlatforms);
	for (unsigned int i = 0; i < numPlatforms; i++) {
		char buf[100];
		printf("Platform %u: \n", i);
		status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
				sizeof(buf), buf, NULL);

		printf("\tVendor: %s\n", buf);

		status |= clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
				sizeof(buf), buf, NULL);
		printf("\tName: %s\n", buf);
		if (status != CL_SUCCESS) {
			printf("clGetPlatformInfo failed\n");
			exit(-1);
		}
	}

	printf("\n");
	cl_uint numDevices = 0;
	cl_device_id *devices;
	// Retrieve the number of devices present

	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL,
			&numDevices);

	if (status != CL_SUCCESS) {
		printf("clGetDeviceIDs failed\n");
		exit(-1);
	}

	// Make sure some devices were found
	if (numDevices == 0) {
		printf("No devices detected.\n");
		exit(-1);
	}

	// Allocate enough space for each device
	devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
	if (devices == NULL) {
		perror("malloc");
		exit(-1);
	}

	// Fill in devices
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices,
			devices, NULL);
	if (status != CL_SUCCESS) {
		printf("clGetDeviceIDs failed\n");
		exit(-1);
	}

	// Print out some basic information about each device

	printf("%u devices detected\n", numDevices);
	for (unsigned int i = 0; i < numDevices; i++) {
		char buf[100];
		printf("Device %u: \n", i);
		status = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buf),
				buf, NULL);
		printf("\tDevice: %s\n", buf);
		status |= clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buf), buf,
				NULL);
		printf("\tName: %s\n", buf);
		if (status != CL_SUCCESS) {
			printf("clGetDeviceInfo failed\n");
			exit(-1);
		}
	}

	printf("\n");

	cl_context context;

	// Create a context and associate it with the devices
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

	if (status != CL_SUCCESS || context == NULL) {
		printf("clCreateContext failed\n");
		exit(-1);
	}

	cl_command_queue cmdQueue;

	// Create a command queue and associate it with the device you
	// want to execute on
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
	if (status != CL_SUCCESS || cmdQueue == NULL) {
		printf("clCreateCommandQueue failed\n");
		exit(-1);
	}
	/////////////////////CREATE BUFFERS HERE//////////////////////////////
	cl_mem xIonKernel; // Input buffers on device
	cl_mem xElectronKernel;
	cl_mem numeroIonesCeldaKernel;
	cl_mem numeroElectronesCeldaKernel;
	cl_mem rhoKernel;
	cl_mem rhoPoissonKernel;
	cl_mem diagonalInferiorKernel;
	cl_mem diagonalSuperiorKernel;
	cl_mem diagonalPrincipalKernel;
	cl_mem phiKernel;
	cl_mem campoElectricoKernel;
	cl_mem campoElectricoIonKernel;
	cl_mem fuerzaParticulaIonKernel;
	cl_mem campoElectricoElectronKernel;
	cl_mem fuerzaParticulaElectronKernel;
	cl_mem vectorContadorIonKernel;
	cl_mem vectorContadorElectronKernel;
	cl_mem velocidadParticulaIonKernel;
	cl_mem velocidadParticulaElectronKernel;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Create a buffer
	xIonKernel = clCreateBuffer(context, CL_MEM_READ_WRITE
			|CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * numeroParticulas, xIon,
			&status);
	if (status != CL_SUCCESS || xIonKernel == NULL) {
		printf("clCreateBuffer failed xIonKernel\n");
		exit(-1);
	}

	// Create a buffer
	xElectronKernel = clCreateBuffer(context, CL_MEM_READ_WRITE
			| CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * numeroParticulas,
			xElectron, &status);
	if (status != CL_SUCCESS || xElectronKernel == NULL) {
		printf("clCreateBuffer failed xElectronKernel\n ");
		exit(-1);
	}
	// Create a buffer
	numeroIonesCeldaKernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float) * numeroParticulas, NULL, &status);
	if (status != CL_SUCCESS || numeroIonesCeldaKernel == NULL) {
		printf("clCreateBuffer failed\n");
		exit(-1);
	}

	// Create a buffer
	numeroElectronesCeldaKernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float) * numeroParticulas, NULL, &status);
	if (status != CL_SUCCESS || numeroElectronesCeldaKernel == NULL) {
		printf("clCreateBuffer failed\n");
		exit(-1);
	}

	// Create a buffer
	rhoKernel = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)
			* numeroMallas, NULL, &status);

	if (status != CL_SUCCESS || rhoKernel == NULL) {
		printf("clCreateBuffer failed rhoKernel\n");
		exit(-1);
	}

	// Create a buffer
	rhoPoissonKernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float) * (numeroMallas - 2), NULL, &status);

	if (status != CL_SUCCESS || rhoPoissonKernel == NULL) {
		printf("clCreateBuffer failed rhoKernel\n");
		exit(-1);
	}

	// Create a buffer
	diagonalInferiorKernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float) * (numeroMallas - 2), NULL, &status);

	if (status != CL_SUCCESS || diagonalInferiorKernel == NULL) {
		printf("clCreateBuffer failed diagonalInferiorKernel\n");
		exit(-1);
	}
	// Create a buffer
	diagonalSuperiorKernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float) * (numeroMallas - 2), NULL, &status);

	if (status != CL_SUCCESS || diagonalSuperiorKernel == NULL) {
		printf("clCreateBuffer failed diagonalSuperiorKernel\n");
		exit(-1);
	}
	// Create a buffer
	diagonalPrincipalKernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float) * (numeroMallas - 2), NULL, &status);

	if (status != CL_SUCCESS || diagonalPrincipalKernel == NULL) {
		printf("clCreateBuffer failed diagonalPrincipalKernel\n");
		exit(-1);
	}
	// Create a buffer
	phiKernel = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)
			* numeroMallas, NULL, &status);

	if (status != CL_SUCCESS || phiKernel == NULL) {
		printf("clCreateBuffer failed phiKernel\n");
		exit(-1);
	}
	// Create a buffer
	campoElectricoKernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float) * numeroMallas, NULL, &status);

	if (status != CL_SUCCESS || campoElectricoKernel == NULL) {
		printf("clCreateBuffer failed campoElectrico\n");
		exit(-1);
	}
	// Create a buffer
	campoElectricoIonKernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float) * numeroParticulas, NULL, &status);

	if (status != CL_SUCCESS || campoElectricoIonKernel == NULL) {
		printf("clCreateBuffer failed campoElectricoIon\n");
		exit(-1);
	}
	// Create a buffer
	fuerzaParticulaIonKernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float) * numeroParticulas, NULL, &status);

	if (status != CL_SUCCESS || fuerzaParticulaIonKernel == NULL) {
		printf("clCreateBuffer failed fuerzaParticulaIon\n");
		exit(-1);
	}

	// Create a buffer
	campoElectricoElectronKernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float) * numeroParticulas, NULL, &status);

	if (status != CL_SUCCESS || campoElectricoElectronKernel == NULL) {
		printf("clCreateBuffer failed campoElectricoElectron\n");
		exit(-1);
	}

	// Create a buffer
	fuerzaParticulaElectronKernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float) * numeroParticulas, NULL, &status);

	if (status != CL_SUCCESS || fuerzaParticulaElectronKernel == NULL) {
		printf("clCreateBuffer failed fuerzaParticulaElectron\n");
		exit(-1);
	}

	// Create a buffer
	vectorContadorIonKernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_int) * numeroParticulas, NULL, &status);

	if (status != CL_SUCCESS || vectorContadorIonKernel == NULL) {
		printf("clCreateBuffer failed vectorContadorIonKernel\n");
		exit(-1);
	}

	vectorContadorElectronKernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_int) * numeroParticulas, NULL, &status);

	if (status != CL_SUCCESS || vectorContadorElectronKernel == NULL) {
		printf("clCreateBuffer failed fuerzaParticulaElectron\n");
		exit(-1);
	}

	velocidadParticulaIonKernel = clCreateBuffer(context, CL_MEM_READ_WRITE
			| CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * numeroParticulas,
			velocidadParticulaIon, &status);

	if (status != CL_SUCCESS || velocidadParticulaIonKernel == NULL) {
		printf("clCreateBuffer failed velocidadParticulaIonKernel \n");
		exit(-1);
	}

	velocidadParticulaElectronKernel = clCreateBuffer(context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float)
					* numeroParticulas, velocidadParticulaElectron, &status);

	if (status != CL_SUCCESS || velocidadParticulaElectronKernel == NULL) {
		printf("clCreateBuffer failed velocidadParticulaElectronKernel\n");
		exit(-1);
	}

	/////////////////////////////////////////////////////////////////////////////////

	cl_program program;
	char *source;
	const char *sourceFile =
			"/home/john/picProject/picProject/parallelPicFinal/src/kernelOpencl";
	// This function reads in the source code of the program

	source = readSource(sourceFile);

	//printf("Program source is:\n%s\n", source);

	// Create a program. The 'source' string is the code from the
	// vectoradd.cl file.
	program = clCreateProgramWithSource(context, 1, (const char**) &source,
			NULL, &status);
	if (status != CL_SUCCESS) {
		printf("clCreateProgramWithSource failed\n");
		exit(-1);
	}

	cl_int buildErr;

	// Build (compile & link) the program for the devices.
	// Save the return value in 'buildErr' (the following
	// code will print any compilation errors to the screen)
	buildErr = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

	// If there are build errors, print them to the screen
	if (buildErr != CL_SUCCESS) {
		printf("Program failed to build.\n");
		cl_build_status buildStatus;
		for (unsigned int i = 0; i < numDevices; i++) {
			clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_STATUS,
					sizeof(cl_build_status), &buildStatus, NULL);
			if (buildStatus == CL_SUCCESS) {
				continue;
			}

			char *buildLog;
			size_t buildLogSize;
			clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, 0,
					NULL, &buildLogSize);
			buildLog = (char*) malloc(buildLogSize);
			if (buildLog == NULL) {
				perror("malloc");
				exit(-1);
			}

			clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
					buildLogSize, buildLog, NULL);

			buildLog[buildLogSize - 1] = '\0';
			printf("Device %u Build Log:\n%s\n", i, buildLog);
			free(buildLog);
		}
		exit(0);
	} else {
		printf("No build errors\n");
	}
	///////////////////////CREATE KERNEL/////////////////////////////////////////
	cl_kernel kernelFuncionNumeroParticulasCelda;
	cl_kernel kernelFuncionRho;
	cl_kernel kernelFuncionPotencialElectroEstatico;
	cl_kernel kernelCalculoCampoElectrico;
	cl_kernel kernelCalculoFuerzaParticulaIon;
	cl_kernel kernelCalculoFuerzaParticulaElectron;
	cl_kernel kernelActualizacionVelocidadPosicion;

	/////////////////////////////////////////////////////////////////////////////////

	kernelFuncionNumeroParticulasCelda = clCreateKernel(program,
			"funcionNumeroParticulasCelda", &status);
	if (status != CL_SUCCESS) {
		printf("clCreateKernel failed funcionNumeroParticulasCelda\n");
		exit(-1);
	}
	// Associate the input and output buffers with the kernel
	status = clSetKernelArg(kernelFuncionNumeroParticulasCelda, 0,
			sizeof(cl_float), &tamanoCelda);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg failed tamanoCelda FuncionNumeroParticulasCelda\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelFuncionNumeroParticulasCelda, 1,
			sizeof(cl_int), &numeroMallas);
	if (status != CL_SUCCESS) {

		printf(
				"clSetKernelArg failed numeroMallas FuncionNumeroParticulasCelda\n");
		exit(-1);
	}
	status = clSetKernelArg(kernelFuncionNumeroParticulasCelda, 2,
			sizeof(cl_int), &numeroParticulas);
	if (status != CL_SUCCESS) {

		printf(
				"clSetKernelArg failed numeroParticulas FuncionNumeroParticulasCelda\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelFuncionNumeroParticulasCelda, 3,
			sizeof(cl_mem), &xIonKernel);
	if (status != CL_SUCCESS) {

		printf("clSetKernelArg failed xIon FuncionNumeroParticulasCelda\n");
		exit(-1);
	}
	status = clSetKernelArg(kernelFuncionNumeroParticulasCelda, 4,
			sizeof(cl_mem), &xElectronKernel);
	if (status != CL_SUCCESS) {

		printf("clSetKernelArg failed xElectron FuncionNumeroParticulasCelda\n");
		exit(-1);
	}
	status = clSetKernelArg(kernelFuncionNumeroParticulasCelda, 5,
			sizeof(cl_mem), &numeroIonesCeldaKernel);
	if (status != CL_SUCCESS) {

		printf(
				"clSetKernelArg failed numeroIonesCelda FuncionNumeroParticulasCelda\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelFuncionNumeroParticulasCelda, 6,
			sizeof(cl_mem), &numeroElectronesCeldaKernel);
	if (status != CL_SUCCESS) {

		printf(
				"clSetKernelArg failed numeroElectronesCelda FuncionNumeroParticulasCelda\n");
		exit(-1);
	}
	///////////////////////////////////////////RH0///////////////////////////////////////////////
	kernelFuncionRho = clCreateKernel(program, "funcionRho", &status);
	if (status != CL_SUCCESS) {
		printf("clCreateKernel failed\n");
		exit(-1);
	}
	status = clSetKernelArg(kernelFuncionRho, 0, sizeof(cl_float), &cargaIon);
	if (status != CL_SUCCESS) {

		printf("clSetKernelArg failed cargaIon FuncionRho\n");
		exit(-1);
	}
	status
			= clSetKernelArg(kernelFuncionRho, 1, sizeof(cl_float),
					&tamanoCelda);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg failed tamanoCelda FuncionRho\n");
		exit(-1);
	}
	status = clSetKernelArg(kernelFuncionRho, 2, sizeof(cl_mem),
			&numeroIonesCeldaKernel);
	if (status != CL_SUCCESS) {

		printf("clSetKernelArg failed numeroIonesCelda FuncionRho\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelFuncionRho, 3, sizeof(cl_mem),
			&numeroElectronesCeldaKernel);
	if (status != CL_SUCCESS) {

		printf("clSetKernelArg failed numeroElectronesCelda FuncionRho\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelFuncionRho, 4, sizeof(cl_mem), &rhoKernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg failed\n");
		exit(-1);
	}
	////////////////////////PHI///////////////////////////////////////////////////
	kernelFuncionPotencialElectroEstatico = clCreateKernel(program,
			"calculoPotencialElectroEstatico", &status);
	if (status != CL_SUCCESS) {
		printf("clCreateKernel calculoPotencialElectroEstatico failed\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelFuncionPotencialElectroEstatico, 0,
			sizeof(cl_float), &permitividadElectrica);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg permitividadElectrica failed\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelFuncionPotencialElectroEstatico, 1,
			sizeof(cl_float), &numeroMallas);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg numeroMallas failed\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelFuncionPotencialElectroEstatico, 2,
			sizeof(cl_float), &tamanoCelda);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg tamanoCelda failed\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelFuncionPotencialElectroEstatico, 3,
			sizeof(cl_mem), &rhoKernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg rhoKernel failed\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelFuncionPotencialElectroEstatico, 4,
			sizeof(cl_mem), &rhoPoissonKernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg rhoPoissonKernel failed\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelFuncionPotencialElectroEstatico, 5,
			sizeof(cl_mem), &diagonalInferiorKernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg diagonalInferiorKernel failed\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelFuncionPotencialElectroEstatico, 6,
			sizeof(cl_mem), &diagonalSuperiorKernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg diagonalSuperiorKernel failed\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelFuncionPotencialElectroEstatico, 7,
			sizeof(cl_mem), &diagonalPrincipalKernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg diagonalPrincipalKernel failed\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelFuncionPotencialElectroEstatico, 8,
			sizeof(cl_mem), &phiKernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg phiKernel failed\n");
		exit(-1);
	}
	///////////////////////////ELECTRIC FIELD/////////////////////////////////
	kernelCalculoCampoElectrico = clCreateKernel(program,
			"calculoCampoElectrico", &status);
	if (status != CL_SUCCESS) {
		printf("clCreateKernel calculoCampoElectrico failed\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelCalculoCampoElectrico, 0, sizeof(cl_float),
			&tamanoCelda);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg tamanoCelda failed kernelCalculoCampoElectrico\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelCalculoCampoElectrico, 1, sizeof(cl_int),
			&numeroMallas);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg numeroMallas failed kernelCalculoCampoElectrico\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelCalculoCampoElectrico, 2, sizeof(cl_mem),
			&phiKernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg phi failed kernelCalculoCampoElectrico\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelCalculoCampoElectrico, 3, sizeof(cl_mem),
			&campoElectricoKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg campoElectrico failed kernelCalculoCampoElectrico\n");
		exit(-1);
	}
	//////////////////////////FUERZA PARTICULA ION ///////////////////////////


	kernelCalculoFuerzaParticulaIon = clCreateKernel(program,
			"calculoFuerzaParticulaIon", &status);
	if (status != CL_SUCCESS) {
		printf("clCreateKernel calculoFuerzaParticulaIon failed\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelCalculoFuerzaParticulaIon, 0, sizeof(cl_int),
			&numeroMallas);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg numeroMallas failed kernelCalculoFuerzaParticulaIon\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelCalculoFuerzaParticulaIon, 1,
			sizeof(cl_float), &tamanoCelda);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg tamanoCelda failed kernelCalculoFuerzaParticulaIon\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelCalculoFuerzaParticulaIon, 2,
			sizeof(cl_float), &xita);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg xita failed kernelCalculoFuerzaParticulaIon\n");
		exit(-1);
	}
	status = clSetKernelArg(kernelCalculoFuerzaParticulaIon, 3, sizeof(cl_mem),
			&campoElectricoKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg campoElectricoKernel failed kernelCalculoFuerzaParticulaIon\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelCalculoFuerzaParticulaIon, 4, sizeof(cl_mem),
			&xIonKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg xIonKernel failed kernelCalculoFuerzaParticulaIon\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelCalculoFuerzaParticulaIon, 5, sizeof(cl_mem),
			&campoElectricoIonKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg campoElectricoIonKernel failed kernelCalculoFuerzaParticulaIon\n");
		exit(-1);
	}
	status = clSetKernelArg(kernelCalculoFuerzaParticulaIon, 6, sizeof(cl_mem),
			&fuerzaParticulaIonKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg fuerzaParticulaIonKernel failed kernelCalculoFuerzaParticulaIon\n");
		exit(-1);
	}
	//////////////////////////FUERZA PARTICULA ELECTRON///////////////////////////
	kernelCalculoFuerzaParticulaElectron = clCreateKernel(program,
			"calculoFuerzaParticulaElectron", &status);
	if (status != CL_SUCCESS) {
		printf("clCreateKernel calculoFuerzaParticulaElectron failed\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelCalculoFuerzaParticulaElectron, 0,
			sizeof(cl_int), &numeroMallas);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg numeroMallas failed kernelCalculoFuerzaParticulaElectron\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelCalculoFuerzaParticulaElectron, 1,
			sizeof(cl_float), &tamanoCelda);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg tamanoCelda failed kernelCalculoFuerzaParticulaElectron\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelCalculoFuerzaParticulaElectron, 2,
			sizeof(cl_float), &xita);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg xita failed kernelCalculoFuerzaParticulaElectron\n");
		exit(-1);
	}
	status = clSetKernelArg(kernelCalculoFuerzaParticulaElectron, 3,
			sizeof(cl_mem), &campoElectricoKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg campoElectricoKernel failed kernelCalculoFuerzaParticulaElectron\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelCalculoFuerzaParticulaElectron, 4,
			sizeof(cl_mem), &xElectronKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg xIonKernel failed kernelCalculoFuerzaParticulaIon\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelCalculoFuerzaParticulaElectron, 5,
			sizeof(cl_mem), &campoElectricoElectronKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg campoElectricoIonKernel failed kernelCalculoFuerzaParticulaElectron\n");
		exit(-1);
	}
	status = clSetKernelArg(kernelCalculoFuerzaParticulaElectron, 6,
			sizeof(cl_mem), &fuerzaParticulaElectronKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg fuerzaParticulaIonKernel failed kernelCalculoFuerzaParticulaElectron\n");
		exit(-1);
	}
	//////////////////////////////ACTUALIZACION VELOCIDAD POSICION///////////////////////////////////////

	kernelActualizacionVelocidadPosicion = clCreateKernel(program,
			"actualizacionVelocidadPosicion", &status);
	if (status != CL_SUCCESS) {
		printf("clCreateKernel actualizacionVelocidadPosicion failed\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelActualizacionVelocidadPosicion, 0,
			sizeof(cl_float), &pasoTiempo);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg pasoTiempo failed kernelActualizacionVelocidadPosicion\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelActualizacionVelocidadPosicion, 1,
			sizeof(cl_int), &numeroMallas);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg numeroMallas failed kernelActualizacionVelocidadPosicion\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelActualizacionVelocidadPosicion, 2,
			sizeof(cl_float), &tamanoCelda);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg tamanoCelda failed kernelActualizacionVelocidadPosicion\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelActualizacionVelocidadPosicion, 3,
			sizeof(cl_mem), &xIonKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg xIonKernel failed kernelActualizacionVelocidadPosicion\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelActualizacionVelocidadPosicion, 4,
			sizeof(cl_mem), &xElectronKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg xElectronKernel failed kernelActualizacionVelocidadPosicion\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelActualizacionVelocidadPosicion, 5,
			sizeof(cl_mem), &fuerzaParticulaIonKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg fuerzaParticulaIonKernel failed kernelActualizacionVelocidadPosicion\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelActualizacionVelocidadPosicion, 6,
			sizeof(cl_mem), &fuerzaParticulaElectronKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg fuerzaParticulaElectronKernel failed kernelActualizacionVelocidadPosicion\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelActualizacionVelocidadPosicion, 7,
			sizeof(cl_mem), &vectorContadorIonKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg vectorContadorIonKernel failed kernelActualizacionVelocidadPosicion\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelActualizacionVelocidadPosicion, 8,
			sizeof(cl_mem), &vectorContadorElectronKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg vectorContadorElectronKernel failed kernelActualizacionVelocidadPosicion\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelActualizacionVelocidadPosicion, 9,
			sizeof(cl_mem), &velocidadParticulaIonKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg vectorParticulaIonKernel failed kernelActualizacionVelocidadPosicion\n");
		exit(-1);
	}

	status = clSetKernelArg(kernelActualizacionVelocidadPosicion, 10,
			sizeof(cl_mem), &velocidadParticulaElectronKernel);
	if (status != CL_SUCCESS) {
		printf(
				"clSetKernelArg velocidadParticulaElectronKernel failed kernelActualizacionVelocidadPosicion\n");
		exit(-1);
	}
	/////////////////////////SIZE KERNEL/////////////////////////////////////////////////////////////////
	size_t global[1];
	global[0] = numeroMallas;

	size_t globalNumeroParticulas[1];
	globalNumeroParticulas[0] = numeroParticulas;

	/////////////////////////CALL KERNEL//////////////////////////////////////////
	start = clock();

	for (int i = 0; i < numeroIteraciones; i++) {
		//////////////////////////////////////////////////////////////////////////////////
		status = clEnqueueNDRangeKernel(cmdQueue,
				kernelFuncionNumeroParticulasCelda, 1, NULL, global, NULL, 0,
				NULL, NULL);
		if (status != CL_SUCCESS) {
			printf(
					"clEnqueueNDRangeKernel kernelFuncionNumeroParticulasCelda failed\n");
			exit(-1);
		}
		///////////////////////////////////////////////////////////////////////////////////
		status = clEnqueueNDRangeKernel(cmdQueue, kernelFuncionRho, 1, NULL,
				global, NULL, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			printf("clEnqueueNDRangeKernel kernelFuncionRho failed\n");
			exit(-1);
		}
		////////////////////////////////////////////////////////////////////////////////////

		status = clEnqueueNDRangeKernel(cmdQueue,
				kernelFuncionPotencialElectroEstatico, 1, NULL, global, NULL,
				0, NULL, NULL);
		if (status != CL_SUCCESS) {
			printf(
					"clEnqueueNDRangeKernel kernelFuncionPotencialElectroEstatico failed\n");
			exit(-1);
		}
		////////////////////////////////////////////////////////////////////////////////////
		status = clEnqueueNDRangeKernel(cmdQueue, kernelCalculoCampoElectrico,
				1, NULL, global, NULL, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			printf(
					"clEnqueueNDRangeKernel kernelCalculoCampoElectrico failed\n");
			exit(-1);
		}
		/////////////////////////////////////////////////////////////////////////////////////
		status = clEnqueueNDRangeKernel(cmdQueue,
				kernelCalculoFuerzaParticulaIon, 1, NULL,
				globalNumeroParticulas, NULL, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			printf(
					"clEnqueueNDRangeKernel kernelCalculoFuerzaParticulaIon failed\n");
			exit(-1);
		}
		/////////////////////////////////////////////////////////////////////////////////////
		status = clEnqueueNDRangeKernel(cmdQueue,
				kernelCalculoFuerzaParticulaElectron, 1, NULL,
				globalNumeroParticulas, NULL, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			printf(
					"clEnqueueNDRangeKernel kernelCalculoFuerzaParticulaElectron failed\n");
			exit(-1);
		}
		/////////////////////////////////////////////////////////////////////////////////////

		status = clEnqueueNDRangeKernel(cmdQueue,
				kernelActualizacionVelocidadPosicion, 1, NULL,
				globalNumeroParticulas, NULL, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			printf(
					"clEnqueueNDRangeKernel kernelActualizacionVelocidadPosicion failed\n");
			exit(-1);
		}
		/////////////////////////////////////////////////////////////////////////////////////

	}
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Time elapsed: %.20f\n", cpu_time_used);

	/////////////////////////READ BUFFER/////////////////////////////////////////
	clEnqueueReadBuffer(cmdQueue, numeroIonesCeldaKernel, CL_TRUE, 0,
			sizeof(float) * numeroMallas, numeroIonesCelda, 0, NULL, NULL);

	clEnqueueReadBuffer(cmdQueue, numeroElectronesCeldaKernel, CL_TRUE, 0,
			sizeof(float) * numeroMallas, numeroElectronesCelda, 0, NULL, NULL);

	clEnqueueReadBuffer(cmdQueue, rhoKernel, CL_TRUE, 0, sizeof(float)
			* numeroMallas, rho, 0, NULL, NULL);

	clEnqueueReadBuffer(cmdQueue, phiKernel, CL_TRUE, 0, sizeof(float)
			* numeroMallas, phi, 0, NULL, NULL);

	clEnqueueReadBuffer(cmdQueue, campoElectricoKernel, CL_TRUE, 0,
			sizeof(float) * numeroMallas, campoElectrico, 0, NULL, NULL);

	clEnqueueReadBuffer(cmdQueue, fuerzaParticulaIonKernel, CL_TRUE, 0,
			sizeof(float) * numeroParticulas, fuerzaParticulaIon, 0, NULL, NULL);

	clEnqueueReadBuffer(cmdQueue, fuerzaParticulaElectronKernel, CL_TRUE, 0,
			sizeof(float) * numeroParticulas, fuerzaParticulaElectron, 0, NULL,
			NULL);

	clEnqueueReadBuffer(cmdQueue, velocidadParticulaElectronKernel, CL_TRUE, 0,
			sizeof(float) * numeroParticulas, velocidadParticulaElectron, 0,
			NULL, NULL);

	clEnqueueReadBuffer(cmdQueue, velocidadParticulaIonKernel, CL_TRUE, 0,
			sizeof(float) * numeroParticulas, velocidadParticulaIon, 0, NULL,
			NULL);

	clEnqueueReadBuffer(cmdQueue, xElectronKernel, CL_TRUE, 0, sizeof(float)
			* numeroParticulas, xElectron, 0, NULL, NULL);

	clEnqueueReadBuffer(cmdQueue, xIonKernel, CL_TRUE, 0, sizeof(float)
			* numeroParticulas, xIon, 0, NULL, NULL);

	//////////////////////////FREE MEMORY////////////////////////////////////////

	clReleaseKernel(kernelFuncionNumeroParticulasCelda);
	clReleaseKernel(kernelFuncionRho);
	clReleaseKernel(kernelFuncionPotencialElectroEstatico);
	clReleaseKernel(kernelCalculoCampoElectrico);
	clReleaseKernel(kernelCalculoFuerzaParticulaIon);
	clReleaseKernel(kernelCalculoFuerzaParticulaElectron);
	clReleaseKernel(kernelActualizacionVelocidadPosicion);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(xIonKernel);
	clReleaseMemObject(xElectronKernel);
	clReleaseMemObject(numeroIonesCeldaKernel);
	clReleaseMemObject(numeroElectronesCeldaKernel);
	clReleaseMemObject(rhoKernel);
	clReleaseMemObject(diagonalInferiorKernel);
	clReleaseMemObject(diagonalSuperiorKernel);
	clReleaseMemObject(diagonalPrincipalKernel);
	clReleaseMemObject(rhoPoissonKernel);
	clReleaseMemObject(phiKernel);
	clReleaseMemObject(campoElectricoKernel);
	clReleaseMemObject(campoElectricoIonKernel);
	clReleaseMemObject(fuerzaParticulaIonKernel);
	clReleaseMemObject(campoElectricoElectronKernel);
	clReleaseMemObject(fuerzaParticulaElectronKernel);
	clReleaseMemObject(vectorContadorElectronKernel);
	clReleaseMemObject(vectorContadorIonKernel);
	clReleaseMemObject(velocidadParticulaElectronKernel);
	clReleaseMemObject(velocidadParticulaIonKernel);
	clReleaseContext(context);
	return 0;
}

int main() {
	////////////////// VARIABLES DECLARATION////////////////////////////////////////////
	float K, T, masaElectron, cargaElectron, permitividadElectrica, masaIon;
	float velocidad, cargaIon, tiempoEvaporacion, flujoEvaporacion;
	float tamanoCelda, pasoTiempo, xita, respuesta, *xIon, *xElectron;
	float *numeroIonesCelda, *numeroElectronesCelda, *rho, *rhoPoisson,
			*diagonalInferior;
	float *diagonalSuperior, *diagonalPrincipal, *phi, *campoElectrico,
			*campoElectricoIon, *fuerzaParticulaIon, *campoElectricoElectron,
			*fuerzaParticulaElectron, *velocidadParticulaElectron,
			*velocidadParticulaIon;
	int numeroParticulas, numeroMallas, i = 0, numeroIteraciones,
			*vectorContadorIon, *vectorContadorElectron, fscanfResult, retorno;

	FILE *fpic;
	char buffer[100];

	/////////////////////////////////////////////////////////////////

	fpic = fopen("/home/john/picProject/picProject/parallelPicFinal/src/Archivo2", "r");

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%f", &K);
	printf("%.29f\n", K);

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%f", &T);
	printf("%.29f\n", T);

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%f", &masaElectron);
	printf("%.29f\n", masaElectron);

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%f", &cargaElectron);
	printf("%.29f\n", cargaElectron);

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%f", &permitividadElectrica);
	printf("%.29f\n", permitividadElectrica);

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%f", &masaIon);
	printf("%.29f\n", masaIon);

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%f", &velocidad);
	printf("%.29f\n", velocidad);

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%f", &cargaIon);
	printf("%.29f\n", cargaIon);

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%f", &tiempoEvaporacion);
	printf("%.29f\n", tiempoEvaporacion);

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%f", &flujoEvaporacion);
	printf("%.29f\n", flujoEvaporacion);

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%f", &tamanoCelda);
	printf("%.29f\n", tamanoCelda);

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%f", &pasoTiempo);
	printf("%.29f\n", pasoTiempo);

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%d", &numeroParticulas);
	printf("%d\n", numeroParticulas);

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%d", &numeroMallas);
	printf("%d\n", numeroMallas);

	fscanfResult = fscanf(fpic, "%s ", buffer);
	printf("%s ", buffer);

	fscanfResult = fscanf(fpic, "%f", &xita);
	printf("%f\n", xita);

	retorno = fclose(fpic);

	numeroIteraciones = 400000;

	////////////////// MEMORY ALLOCATION/////////////////////////////////////////////////
	numeroIonesCelda = (float *) malloc(numeroMallas * sizeof(float));
	if (numeroIonesCelda == NULL) {
		printf("No se pudo reservar memoria para la variable numeroIonesCelda");
		return -1;
	}

	numeroElectronesCelda = (float *) malloc(numeroMallas * sizeof(float));
	if (numeroIonesCelda == NULL) {
		printf(
				"No se pudo reservar memoria para la variable numeroElectronesCelda");
		return -1;
	}

	xIon = (float *) malloc(numeroParticulas * sizeof(float));
	if (xIon == NULL) {
		printf("No se pudo reservar memoria para la variable xIon");
		return -1;
	}

	xElectron = (float *) malloc(numeroParticulas * sizeof(float));
	if (xElectron == NULL) {
		printf("No se pudo reservar memoria para la variable xElectron");
		return -1;
	}
	rho = (float *) malloc(numeroMallas * sizeof(float));
	if (rho == NULL) {
		printf("No se pudo reservar memoria para la variable rho");
		return -1;
	}
	rhoPoisson = (float *) malloc((numeroMallas - 2) * sizeof(float));
	if (rhoPoisson == NULL) {
		printf("No se pudo reservar memoria para la variable rhoPoisson");
		return -1;
	}
	phi = (float *) malloc((numeroMallas) * sizeof(float));
	if (phi == NULL) {
		printf("No se pudo reservar memoria para la variable phi");
		return -1;
	}
	diagonalInferior = (float *) malloc((numeroMallas - 2) * sizeof(float));
	if (diagonalInferior == NULL) {
		printf("No se pudo reservar memoria para la variable diagonalInferior");
		return -1;
	}
	diagonalSuperior = (float *) malloc((numeroMallas - 2) * sizeof(float));
	if (diagonalSuperior == NULL) {
		printf("No se pudo reservar memoria para la variable diagonalSuperior");
		return -1;
	}

	diagonalPrincipal = (float *) malloc((numeroMallas - 2) * sizeof(float));
	if (diagonalPrincipal == NULL) {
		printf("No se pudo reservar memoria para la variable diagonalPrincipal");
		return -1;
	}
	campoElectrico = (float *) malloc(numeroMallas * sizeof(float));
	if (campoElectrico == NULL) {
		printf("No se pudo reservar memoria para la variable campoElectrico");
		return -1;
	}
	campoElectricoIon = (float *) malloc(numeroParticulas * sizeof(float));
	if (campoElectricoIon == NULL) {
		printf("No se pudo reservar memoria para la variable campoElectricoIon");
		return -1;
	}
	fuerzaParticulaIon = (float *) malloc(numeroParticulas * sizeof(float));
	if (fuerzaParticulaIon == NULL) {
		printf(
				"No se pudo reservar memoria para la variable fuerzaParticulaIon");
		return -1;
	}

	campoElectricoElectron = (float *) malloc(numeroParticulas * sizeof(float));
	if (campoElectricoElectron == NULL) {
		printf(
				"No se pudo reservar memoria para la variable campoElectricoElectron");
		return -1;
	}

	fuerzaParticulaElectron
			= (float *) malloc(numeroParticulas * sizeof(float));
	if (fuerzaParticulaElectron == NULL) {
		printf(
				"No se pudo reservar memoria para la variable fuerzaParticulaElectron");
		return -1;
	}

	vectorContadorElectron = (int *) malloc(numeroParticulas * sizeof(int));
	if (vectorContadorElectron == NULL) {
		printf(
				"No se pudo reservar memoria para la variable vectorContadorElectron");
		return -1;
	}

	vectorContadorIon = (int *) malloc(numeroParticulas * sizeof(int));
	if (vectorContadorIon == NULL) {
		printf("No se pudo reservar memoria para la variable vectorContadorIon");
		return -1;
	}

	velocidadParticulaElectron = (float *) malloc(numeroParticulas
			* sizeof(float));
	if (velocidadParticulaElectron == NULL) {
		printf(
				"No se pudo reservar memoria para la variable velocidadParticulaElectron");
		return -1;
	}

	velocidadParticulaIon = (float *) malloc(numeroParticulas * sizeof(float));
	if (velocidadParticulaIon == NULL) {
		printf(
				"No se pudo reservar memoria para la variable velocidadParticulaIon");
		return -1;
	}

	///////////////// CALL FUNCTION //////////////////////////////////////////////////////
	respuesta = aleatorioX(numeroParticulas, tamanoCelda, xIon, xElectron);
	respuesta = aleatorioVelocidad(numeroParticulas, velocidadParticulaIon,
			velocidadParticulaElectron);

	respuesta = prepararOpenCL(tamanoCelda, numeroMallas, numeroParticulas,
			xIon, xElectron, numeroIonesCelda, numeroElectronesCelda,
			numeroIteraciones, cargaIon, rho, permitividadElectrica,
			rhoPoisson, diagonalInferior, diagonalSuperior, diagonalPrincipal,
			phi, campoElectrico, campoElectricoIon, fuerzaParticulaIon, xita,
			campoElectricoElectron, fuerzaParticulaElectron, vectorContadorIon,
			vectorContadorElectron, velocidadParticulaIon,
			velocidadParticulaElectron, pasoTiempo);
	for (i = 0; i < 20; i++)
		printf("posicionElectron %.29f\n", xElectron[i]);

	//////////////////////////FREE MEMORY////////////////////////////////////////
	free(xIon);
	free(xElectron);
	free(numeroIonesCelda);
	free(numeroElectronesCelda);
	free(rho);
	free(rhoPoisson);
	free(diagonalPrincipal);
	free(diagonalSuperior);
	free(diagonalInferior);
	free(phi);
	free(campoElectrico);
	free(campoElectricoIon);
	free(fuerzaParticulaIon);
	free(campoElectricoElectron);
	free(fuerzaParticulaElectron);
	free(vectorContadorElectron);
	free(vectorContadorIon);
	free(velocidadParticulaElectron);
	free(velocidadParticulaIon);
	return 0;
}
