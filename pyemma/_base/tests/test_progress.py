'''
Created on 29.07.2015

@author: marscher
'''
import unittest
from pyemma._base.progress import ProgressReporter
from pyemma._base.progress.bar import ProgressBar


class TestProgress(unittest.TestCase):

    # FIXME: does not work with nose (because nose already captures stdout)
    """
    def test_silenced(self):
        reporter = ProgressReporter()
        reporter._register(1)
        reporter.silence_progress = True

        from StringIO import StringIO

        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            # call the update method to potentially create output
            reporter._update(1)
            output = out.getvalue().strip()
            # in silence mode we do not want any output!
            assert output == ''
        finally:
            sys.stdout = saved_stdout

    def test_not_silenced(self):
        reporter = ProgressReporter()
        reporter._register(1)
        reporter.silence_progress = False

        from StringIO import StringIO

        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            # call the update method to potentially create output
            reporter._update(1)
            output = out.getvalue().strip()
            # in silence mode we do not want any output!
            print output
            assert output is not ''
        finally:
            sys.stdout = saved_stdout
    """

    def test_callback(self):

        self.has_been_called = 0

        def call_back(stage, progressbar, *args, **kw):
            global has_been_called
            self.has_been_called += 1
            assert isinstance(stage, int)
            assert isinstance(progressbar, ProgressBar)

        class Worker(ProgressReporter):

            def work(self, count):
                self._progress_register(
                    count, description="hard working", stage=0)
                for _ in range(count):
                    self._progress_update(1, stage=0)

        worker = Worker()
        worker.register_progress_callback(call_back, stage=0)
        expected_calls = 100
        worker.work(count=expected_calls)
        self.assertEqual(self.has_been_called, expected_calls)

if __name__ == "__main__":
    unittest.main()
