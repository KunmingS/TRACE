"""Compatibility shim — imports from trace_tad.server.app.

This file exists so that `uvicorn backend.app:app` still works during
development (when cwd is trace-annotator/).  The real code lives in
trace_tad/server/app.py.
"""
from trace_tad.server.app import app  # noqa: F401

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("trace_tad.server.app:app", host='0.0.0.0', port=8000, reload=True)
