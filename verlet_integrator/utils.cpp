#include "utils.h"

void print(float3 v){
	printf("(%f, %f, %f)", v.x, v.y, v.z);
}

//Round a / b to nearest higher integer value
int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}


const char *textFileRead(const char *fn)
{
	/* Note: the `fatalError' thing is a bit of a hack, The proper course of
	* action would be to have people check the return values of textFileRead,
	* but lets avoid cluttering the lab-code even further.
	*/

	FILE *fp;
	char *content = NULL;
	int count = 0;

	if (fn != NULL)
	{
		fp = fopen(fn, "rt");
		if (fp != NULL)
		{
			fseek(fp, 0, SEEK_END);
			count = ftell(fp);
			fseek(fp, 0, SEEK_SET);

			if (count > 0)
			{
				content = new char[count + 1];
				count = fread(content, sizeof(char), count, fp);
				content[count] = '\0';
			}
			else
			{

				char buffer[256];
				printf(buffer, "File '%s' is empty\n", fn);
			}

			fclose(fp);
		}
		else
		{
			char buffer[256];
			printf(buffer, "Unable to read file '%s'\n", fn);
		}
	}
	else
	{

		printf("textFileRead - argument NULL\n");
	}

	return content;
}

GLuint loadShaders(){

	///////////////////////////////////////////////////////////////////////////
	// Create shaders
	///////////////////////////////////////////////////////////////////////////	

	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	// Invoke helper functions (in glutil.h/cpp) to load text files for vertex and fragment shaders.
	const char *vs = textFileRead("simple.vert");
	const char *fs = textFileRead("sphere.frag");

	glShaderSource(vertexShader, 1, &vs, NULL);
	glShaderSource(fragmentShader, 1, &fs, NULL);

	// we are now done with the source and can free the file data, textFileRead uses new [] to.
	// allocate the memory so we must free it using delete [].
	delete[] vs;
	delete[] fs;

	// Compile the shader, translates into internal representation and checks for errors.
	glCompileShader(vertexShader);
	int compileOK;
	// check for compiler errors in vertex shader.
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &compileOK);
	if (!compileOK) {
		printf("error compiling vertex shader");
		return 0;
	}

	// Compile the shader, translates into internal representation and checks for errors.
	glCompileShader(fragmentShader);
	// check for compiler errors in fragment shader.
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &compileOK);
	if (!compileOK) {
		printf("error compiling fragment shader");
		return 0;
	}

	// Create a program object and attach the two shaders we have compiled, the program object contains
	// both vertex and fragment shaders as well as information about uniforms and attributes common to both.
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, fragmentShader);
	glAttachShader(shaderProgram, vertexShader);

	// Now that the fragment and vertex shader has been attached, we no longer need these two separate objects and should delete them.
	// The attachment to the shader program will keep them alive, as long as we keep the shaderProgram.
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);



	// Link the different shaders that are bound to this program, this creates a final shader that 
	// we can use to render geometry with.
	glLinkProgram(shaderProgram);


	// Check for linker errors, many errors, such as mismatched in and out variables between 
	// vertex/fragment shaders,  do not appear before linking.
	{
		GLint linkOk = 0;
		glGetProgramiv(shaderProgram, GL_LINK_STATUS, &linkOk);
		if (!linkOk)
		{
			printf("Linker error");
		}
	}

	return shaderProgram;
}