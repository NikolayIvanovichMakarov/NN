char s_str_error_string[80] = {"No error!"};
char s_str_buf_error_string[80] = {""};

char *get_str_error()
{
	strcpy(s_str_error_string, s_str_buf_error_string);
	/*!
	 * we return not a source of error string, because we need to safe instance of string
	 */
	return s_str_buf_error_string;
}