#   Copyright 2022 Velexi Corporation
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
Tests that package dependencies for `dermaml` are functional
"""

# --- Imports


# --- Tests


def test_imports():
    """
    Test import of problem packages.
    """
    # PyCaret
    try:
        from pycaret import regression  # noqa
    except ImportError as error:
        assert False, f"Failed to import `pycaret`.\n  Error message: {error}"

    # rembg
    try:
        import rembg  # noqa
    except ImportError as error:
        assert False, f"Failed to import `rembg`.\n  Error message: {error}"

    # mediapipe
    try:
        import mediapipe  # noqa
    except ImportError as error:
        assert False, f"Failed to import `mediapipe`.\n  Error message: {error}"
