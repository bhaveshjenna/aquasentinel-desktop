"""View panels for the desktop client.

Each panel builds its plot items once in ``__init__`` and exposes an ``update(engine)`` that
only moves data. Panels are updated only while visible - the main window drives the current
view and leaves the others alone, which is the native equivalent of the Streamlit app dropping
``st.tabs`` for a single-view segmented control.
"""
