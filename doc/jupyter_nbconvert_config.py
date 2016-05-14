# Configuration file for jupyter-nbconvert.

#------------------------------------------------------------------------------
# Configurable configuration
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# LoggingConfigurable configuration
#------------------------------------------------------------------------------

# A parent class for Configurables that log.
#
# Subclasses have a log trait, and the default behavior is to get the logger
# from the currently running Application.

#------------------------------------------------------------------------------
# SingletonConfigurable configuration
#------------------------------------------------------------------------------

# A configurable that only allows one instance.
#
# This class is for classes that should only have one instance of itself or
# *any* subclass. To create and retrieve such a class use the
# :meth:`SingletonConfigurable.instance` method.

#------------------------------------------------------------------------------
# Application configuration
#------------------------------------------------------------------------------

# This is an application.

# The date format used by logging formatters for %(asctime)s
# c.Application.log_datefmt = '%Y-%m-%d %H:%M:%S'

# The Logging format template
# c.Application.log_format = '[%(name)s]%(highlevel)s %(message)s'

# Set the log level by value or name.
# c.Application.log_level = 30

#------------------------------------------------------------------------------
# JupyterApp configuration
#------------------------------------------------------------------------------

# Base class for Jupyter applications

# Answer yes to any prompts.
# c.JupyterApp.answer_yes = False

# Full path of a config file.
# c.JupyterApp.config_file = u''

# Specify a config file to load.
# c.JupyterApp.config_file_name = u''

# Generate default config file.
# c.JupyterApp.generate_config = False

#------------------------------------------------------------------------------
# NbConvertApp configuration
#------------------------------------------------------------------------------

# This application is used to convert notebook files (*.ipynb) to various other
# formats.
#
# WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.

# The export format to be used, either one of the built-in formats, or a dotted
# object name that represents the import path for an `Exporter` class
c.NbConvertApp.export_format = 'rst'

# read a single notebook from stdin.
# c.NbConvertApp.from_stdin = False

# List of notebooks to convert. Wildcards are supported. Filenames passed
# positionally will be added to the list.
# c.NbConvertApp.notebooks = []

# overwrite base name use for output files. can only be used when converting one
# notebook at a time.
# c.NbConvertApp.output_base = ''

# PostProcessor class used to write the results of the conversion
# c.NbConvertApp.postprocessor_class = u''

# Whether to apply a suffix prior to the extension (only relevant when
# converting to notebook format). The suffix is determined by the exporter, and
# is usually '.nbconvert'.
# c.NbConvertApp.use_output_suffix = True

# Writer class used to write the  results of the conversion
# c.NbConvertApp.writer_class = 'FilesWriter'

#------------------------------------------------------------------------------
# NbConvertBase configuration
#------------------------------------------------------------------------------

# Global configurable class for shared config
#
# Useful for display data priority that might be used by many transformers

# DEPRECATED default highlight language, please use language_info metadata
# instead
# c.NbConvertBase.default_language = 'ipython'

# An ordered list of preferred output type, the first encountered will usually
# be used when converting discarding the others.
# c.NbConvertBase.display_data_priority = ['text/html', 'application/pdf', 'text/latex', 'image/svg+xml', 'image/png', 'image/jpeg', 'text/markdown', 'text/plain']

#------------------------------------------------------------------------------
# Exporter configuration
#------------------------------------------------------------------------------

# Class containing methods that sequentially run a list of preprocessors on a
# NotebookNode object and then return the modified NotebookNode object and
# accompanying resources dict.

# List of preprocessors available by default, by name, namespace,  instance, or
# type.
# c.Exporter.default_preprocessors = ['nbconvert.preprocessors.ClearOutputPreprocessor', 'nbconvert.preprocessors.ExecutePreprocessor', 'nbconvert.preprocessors.coalesce_streams', 'nbconvert.preprocessors.SVG2PDFPreprocessor', 'nbconvert.preprocessors.CSSHTMLHeaderPreprocessor', 'nbconvert.preprocessors.LatexPreprocessor', 'nbconvert.preprocessors.HighlightMagicsPreprocessor', 'nbconvert.preprocessors.ExtractOutputPreprocessor']

# Extension of the file that should be written to disk
# c.Exporter.file_extension = '.txt'

# List of preprocessors, by name or namespace, to enable.
# c.Exporter.preprocessors = []

#------------------------------------------------------------------------------
# TemplateExporter configuration
#------------------------------------------------------------------------------

# Exports notebooks into other file formats.  Uses Jinja 2 templating engine to
# output new formats.  Inherit from this class if you are creating a new
# template type along with new filters/preprocessors.  If the filters/
# preprocessors provided by default suffice, there is no need to inherit from
# this class.  Instead, override the template_file and file_extension traits via
# a config file.
#
# - add_anchor - add_prompts - ansi2html - ansi2latex - ascii_only -
# citation2latex - comment_lines - escape_latex - filter_data_type - get_lines -
# get_metadata - highlight2html - highlight2latex - html2text - indent -
# ipython2python - markdown2html - markdown2latex - markdown2rst - path2url -
# posix_path - prevent_list_blocks - strip_ansi - strip_dollars -
# strip_files_prefix - wrap_text

# Dictionary of filters, by name and namespace, to add to the Jinja environment.
# c.TemplateExporter.filters = {}

# formats of raw cells to be included in this Exporter's output.
# c.TemplateExporter.raw_mimetypes = []

#
# c.TemplateExporter.template_extension = '.tpl'

# Name of the template file to use
# c.TemplateExporter.template_file = u''

#
# c.TemplateExporter.template_path = ['.']

#------------------------------------------------------------------------------
# HTMLExporter configuration
#------------------------------------------------------------------------------

# Exports a basic HTML document.  This exporter assists with the export of HTML.
# Inherit from it if you are writing your own HTML template and need custom
# preprocessors/filters.  If you don't need custom preprocessors/ filters, just
# change the 'template_file' config option.

#------------------------------------------------------------------------------
# LatexExporter configuration
#------------------------------------------------------------------------------

# Exports to a Latex template.  Inherit from this class if your template is
# LaTeX based and you need custom tranformers/filters.  Inherit from it if  you
# are writing your own HTML template and need custom tranformers/filters.   If
# you don't need custom tranformers/filters, just change the  'template_file'
# config option.  Place your template in the special "/latex"  subfolder of the
# "../templates" folder.

#
# c.LatexExporter.template_extension = '.tplx'

#------------------------------------------------------------------------------
# MarkdownExporter configuration
#------------------------------------------------------------------------------

# Exports to a markdown document (.md)

#------------------------------------------------------------------------------
# NotebookExporter configuration
#------------------------------------------------------------------------------

# Exports to an IPython notebook.

# The nbformat version to write. Use this to downgrade notebooks.
# c.NotebookExporter.nbformat_version = 4

#------------------------------------------------------------------------------
# PDFExporter configuration
#------------------------------------------------------------------------------

# Writer designed to write to PDF files

# Shell command used to run bibtex.
# c.PDFExporter.bib_command = [u'bibtex', u'{filename}']

# Shell command used to compile latex.
# c.PDFExporter.latex_command = [u'pdflatex', u'{filename}']

# How many times latex will be called.
# c.PDFExporter.latex_count = 3

# File extensions of temp files to remove after running.
# c.PDFExporter.temp_file_exts = ['.aux', '.bbl', '.blg', '.idx', '.log', '.out']

# Whether to display the output of latex commands.
# c.PDFExporter.verbose = False

#------------------------------------------------------------------------------
# PythonExporter configuration
#------------------------------------------------------------------------------

# Exports a Python code file.

#------------------------------------------------------------------------------
# RSTExporter configuration
#------------------------------------------------------------------------------

# Exports restructured text documents.

#------------------------------------------------------------------------------
# ScriptExporter configuration
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SlidesExporter configuration
#------------------------------------------------------------------------------

# Exports HTML slides with reveal.js

# The URL prefix for reveal.js. This can be a a relative URL for a local copy of
# reveal.js, or point to a CDN.
#
# For speaker notes to work, a local reveal.js prefix must be used.
# c.SlidesExporter.reveal_url_prefix = u''

#------------------------------------------------------------------------------
# Preprocessor configuration
#------------------------------------------------------------------------------

# A configurable preprocessor
#
# Inherit from this class if you wish to have configurability for your
# preprocessor.
#
# Any configurable traitlets this class exposed will be configurable in profiles
# using c.SubClassName.attribute = value
#
# you can overwrite :meth:`preprocess_cell` to apply a transformation
# independently on each cell or :meth:`preprocess` if you prefer your own logic.
# See corresponding docstring for informations.
#
# Disabled by default and can be enabled via the config by
#     'c.YourPreprocessorName.enabled = True'

#
# c.Preprocessor.enabled = False

#------------------------------------------------------------------------------
# CSSHTMLHeaderPreprocessor configuration
#------------------------------------------------------------------------------

# Preprocessor used to pre-process notebook for HTML output.  Adds IPython
# notebook front-end CSS and Pygments CSS to HTML output.

# CSS highlight class identifier
# c.CSSHTMLHeaderPreprocessor.highlight_class = '.highlight'

#------------------------------------------------------------------------------
# ClearOutputPreprocessor configuration
#------------------------------------------------------------------------------

# Removes the output from all code cells in a notebook.

#------------------------------------------------------------------------------
# ConvertFiguresPreprocessor configuration
#------------------------------------------------------------------------------

# Converts all of the outputs in a notebook from one format to another.

# Format the converter accepts
# c.ConvertFiguresPreprocessor.from_format = u''

# Format the converter writes
# c.ConvertFiguresPreprocessor.to_format = u''

#------------------------------------------------------------------------------
# ExecutePreprocessor configuration
#------------------------------------------------------------------------------

# Executes all the cells in a notebook

# If `False` (default), when a cell raises an error the execution is stoppped
# and a `CellExecutionError` is raised. If `True`, execution errors are ignored
# and the execution is continued until the end of the notebook. Output from
# exceptions is included in the cell output in both cases.
# c.ExecutePreprocessor.allow_errors = False

# If execution of a cell times out, interrupt the kernel and continue executing
# other cells rather than throwing an error and stopping.
# c.ExecutePreprocessor.interrupt_on_timeout = False

# Name of kernel to use to execute the cells. If not set, use the kernel_spec
# embedded in the notebook.
# c.ExecutePreprocessor.kernel_name = ''

# If `False` (default), then the kernel will continue waiting for iopub messages
# until it receives a kernel idle message, or until a timeout occurs, at which
# point the currently executing cell will be skipped. If `True`, then an error
# will be raised after the first timeout. This option generally does not need to
# be used, but may be useful in contexts where there is the possibility of
# executing notebooks with memory-consuming infinite loops.
# c.ExecutePreprocessor.raise_on_iopub_timeout = False

# The time to wait (in seconds) for output from executions. If a cell execution
# takes longer, an exception (TimeoutError on python 3+, RuntimeError on python
# 2) is raised.
#
# `None` or `-1` will disable the timeout.
c.ExecutePreprocessor.timeout = -1

#------------------------------------------------------------------------------
# ExtractOutputPreprocessor configuration
#------------------------------------------------------------------------------

# Extracts all of the outputs from the notebook file.  The extracted  outputs
# are returned in the 'resources' dictionary.

#
# c.ExtractOutputPreprocessor.extract_output_types = set(['image/png', 'application/pdf', 'image/jpeg', 'image/svg+xml'])

#
# c.ExtractOutputPreprocessor.output_filename_template = '{unique_key}_{cell_index}_{index}{extension}'

#------------------------------------------------------------------------------
# HighlightMagicsPreprocessor configuration
#------------------------------------------------------------------------------

# Detects and tags code cells that use a different languages than Python.

# Syntax highlighting for magic's extension languages. Each item associates a
# language magic extension such as %%R, with a pygments lexer such as r.
# c.HighlightMagicsPreprocessor.languages = {}

#------------------------------------------------------------------------------
# LatexPreprocessor configuration
#------------------------------------------------------------------------------

# Preprocessor for latex destined documents.
#
# Mainly populates the `latex` key in the resources dict, adding definitions for
# pygments highlight styles.

#------------------------------------------------------------------------------
# SVG2PDFPreprocessor configuration
#------------------------------------------------------------------------------

# Converts all of the outputs in a notebook from SVG to PDF.

# The command to use for converting SVG to PDF
#
# This string is a template, which will be formatted with the keys to_filename
# and from_filename.
#
# The conversion call must read the SVG from {from_flename}, and write a PDF to
# {to_filename}.
# c.SVG2PDFPreprocessor.command = u''

# The path to Inkscape, if necessary
# c.SVG2PDFPreprocessor.inkscape = u''

#------------------------------------------------------------------------------
# WriterBase configuration
#------------------------------------------------------------------------------

# Consumes output from nbconvert export...() methods and writes to a useful
# location.

# List of the files that the notebook references.  Files will be  included with
# written output.
# c.WriterBase.files = []

#------------------------------------------------------------------------------
# DebugWriter configuration
#------------------------------------------------------------------------------

# Consumes output from nbconvert export...() methods and writes usefull
# debugging information to the stdout.  The information includes a list of
# resources that were extracted from the notebook(s) during export.

#------------------------------------------------------------------------------
# FilesWriter configuration
#------------------------------------------------------------------------------

# Consumes nbconvert output and produces files.

# Directory to write output to.  Leave blank to output to the current directory
c.FilesWriter.build_directory = 'source/generated/'

# When copying files that the notebook depends on, copy them in relation to this
# path, such that the destination filename will be os.path.relpath(filename,
# relpath). If FilesWriter is operating on a notebook that already exists
# elsewhere on disk, then the default will be the directory containing that
# notebook.
# c.FilesWriter.relpath = ''

#------------------------------------------------------------------------------
# StdoutWriter configuration
#------------------------------------------------------------------------------

# Consumes output from nbconvert export...() methods and writes to the  stdout
# stream.

#------------------------------------------------------------------------------
# PostProcessorBase configuration
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# ServePostProcessor configuration
#------------------------------------------------------------------------------

# Post processor designed to serve files
#
# Proxies reveal.js requests to a CDN if no local reveal.js is present

# The IP address to listen on.
# c.ServePostProcessor.ip = '127.0.0.1'

# Should the browser be opened automatically?
# c.ServePostProcessor.open_in_browser = True

# port for the server to listen on.
# c.ServePostProcessor.port = 8000

# URL for reveal.js CDN.
# c.ServePostProcessor.reveal_cdn = 'https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.1.0'

# URL prefix for reveal.js
# c.ServePostProcessor.reveal_prefix = 'reveal.js'
