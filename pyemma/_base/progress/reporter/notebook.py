from tqdm._tqdm_notebook import tqdm_notebook


# we just override the default formatting of the widget here
class my_tqdm_notebook(tqdm_notebook):

    @staticmethod
    def status_printer(_, total=None, desc=None):
        # Prepare IPython progress bar
        from ipywidgets import IntProgress, HTML, HBox, Layout, Label, Box

        if total:
            pbar = IntProgress(min=0, max=total)
        else:  # No total? Show info style bar with no progress tqdm status
            pbar = IntProgress(min=0, max=1)
            pbar.value = 1
            pbar.bar_style = 'info'
        if desc:
            description = Label(desc, layout=Layout(min_width='20%', max_width='35%'))
        else:
            description = None

        # Prepare status text
        ptext = HTML()
        inner = Box([pbar, ptext], layout=Layout(display='flex'))
        # Only way to place text to the right of the bar is to use a container
        box_layout = Layout(display='flex',
                            justify_content='space-between',
                            width='100%')
        container = HBox(children=[description,
                                   inner], layout=box_layout)
        from IPython.core.display import display
        display(container)

        def print_status(s='', close=False, bar_style=None):
            # Note: contrary to native tqdm, s='' does NOT clear bar
            # goal is to keep all infos if error happens so user knows
            # at which iteration the loop failed.

            # Clear previous output (really necessary?)
            # clear_output(wait=1)

            # Get current iteration value from format_meter string
            if total:
                n = None
                if s:
                    npos = s.find(r'/|/')  # cause we use bar_format=r'{n}|...'
                    # Check that n can be found in s (else n > total)
                    if npos >= 0:
                        n = int(float((s[:npos])))  # get n from string
                        s = s[npos + 3:]  # remove from string

                        # Update bar with current n value
                        if n is not None:
                            pbar.value = n

            # Print stats
            if s:  # never clear the bar (signal: s='')
                s = s.replace('||', '')  # remove inesthetical pipes
                from html import escape
                s = escape(s)  # html escape special characters (like '?')
                ptext.value = s

            # Change bar style
            if bar_style:
                # Hack-ish way to avoid the danger bar_style being overriden by
                # success because the bar gets closed after the error...
                if not (pbar.bar_style == 'danger' and bar_style == 'success'):
                    pbar.bar_style = bar_style

            # Special signal to close the bar
            if close and pbar.bar_style != 'danger':  # hide only if no error
                try:
                    container.close()
                except AttributeError:
                    container.visible = False

        return print_status
