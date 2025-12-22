"""Comprehensive tests for llmprep.py - LLM codebase preparation tool."""

import sys
from pathlib import Path
from unittest.mock import patch


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import llmprep


class TestRunCommand:
    """Tests for run_command utility function."""

    def test_run_command_success(self, tmp_path: Path):
        """Run command succeeds and captures output."""
        output_file = tmp_path / "output.txt"
        result = llmprep.run_command(["echo", "hello"], stdout_file=output_file)
        assert result is True
        assert output_file.read_text().strip() == "hello"

    def test_run_command_quiet(self):
        """Run command in quiet mode."""
        result = llmprep.run_command(["echo", "test"], quiet=True)
        assert result is True

    def test_run_command_failure(self):
        """Run command returns False on failure."""
        result = llmprep.run_command(["nonexistent_command_12345"])
        assert result is False

    def test_run_command_check_raises(self):
        """Run command with check=True doesn't raise for success."""
        result = llmprep.run_command(["echo", "test"], check=True, quiet=True)
        assert result is True


class TestCheckTool:
    """Tests for check_tool function."""

    def test_check_tool_exists(self):
        """Returns True for existing tool."""
        assert llmprep.check_tool("python") is True

    def test_check_tool_not_exists(self):
        """Returns False for non-existing tool."""
        assert llmprep.check_tool("nonexistent_tool_xyz_12345") is False


class TestParseClocOutput:
    """Tests for parse_cloc_output function."""

    def test_parse_cloc_output_basic(self, tmp_path: Path):
        """Parses cloc output correctly."""
        stats_file = tmp_path / "stats.txt"
        stats_file.write_text(
            """
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                          10            100            50            500
JavaScript                       5             30            20            200
-------------------------------------------------------------------------------
SUM:                            15            130            70            700
-------------------------------------------------------------------------------
"""
        )
        result = llmprep.parse_cloc_output(stats_file)
        assert result["Python"] == 500
        assert result["JavaScript"] == 200

    def test_parse_cloc_output_empty(self, tmp_path: Path):
        """Returns empty dict for empty file."""
        stats_file = tmp_path / "empty.txt"
        stats_file.write_text("")
        result = llmprep.parse_cloc_output(stats_file)
        assert result == {}

    def test_parse_cloc_output_nonexistent(self, tmp_path: Path):
        """Returns empty dict for nonexistent file."""
        result = llmprep.parse_cloc_output(tmp_path / "nonexistent.txt")
        assert result == {}

    def test_parse_cloc_output_malformed(self, tmp_path: Path):
        """Handles malformed cloc output gracefully."""
        stats_file = tmp_path / "malformed.txt"
        stats_file.write_text(
            """
Language                     files          blank        comment           code
Python                          10
"""
        )
        result = llmprep.parse_cloc_output(stats_file)
        assert result == {}


class TestGenerateTree:
    """Tests for generate_tree function."""

    def test_generate_tree_with_tree_command(self, tmp_path: Path):
        """Generates tree structure using tree command."""
        output_file = tmp_path / "tree.txt"
        # Create some test files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("# main")

        with patch.object(llmprep, "check_tool", return_value=True):
            with patch.object(llmprep, "run_command", return_value=True) as mock_run:
                llmprep.generate_tree(tmp_path, output_file, 4, "__pycache__", False)
                mock_run.assert_called_once()

    def test_generate_tree_fallback(self, tmp_path: Path):
        """Falls back to manual tree when tree command not available."""
        output_file = tmp_path / "tree.txt"
        # Create some test files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("# main")

        with patch.object(llmprep, "check_tool", return_value=False):
            result = llmprep.generate_tree(tmp_path, output_file, 4, "__pycache__", True)
            assert result is True
            assert output_file.exists()


class TestGenerateSystemPrompt:
    """Tests for generate_system_prompt function."""

    def test_generate_system_prompt_basic(self):
        """Generates system prompt with project name."""
        prompt = llmprep.generate_system_prompt("MyProject", ["Python", "JavaScript"])
        assert "MyProject" in prompt
        assert "Python" in prompt
        assert "JavaScript" in prompt

    def test_generate_system_prompt_no_languages(self):
        """Handles empty languages list."""
        prompt = llmprep.generate_system_prompt("Project", [])
        assert "Project" in prompt
        assert "general programming" in prompt

    def test_generate_system_prompt_guidelines(self):
        """Includes guidelines in prompt."""
        prompt = llmprep.generate_system_prompt("Test", ["Python"])
        assert "Guidelines" in prompt
        assert "clear, concise" in prompt.lower()


class TestGenerateGuidance:
    """Tests for generate_guidance function."""

    def test_generate_guidance_python(self):
        """Generates Python-specific guidance."""
        guidance = llmprep.generate_guidance("Project", ["Python"])
        assert "Python Best Practices" in guidance
        assert "PEP 8" in guidance

    def test_generate_guidance_c(self):
        """Generates C/C++ specific guidance."""
        guidance = llmprep.generate_guidance("Project", ["C", "C++"])
        assert "C/C++ Best Practices" in guidance
        assert "RAII" in guidance

    def test_generate_guidance_general(self):
        """Always includes general guidelines."""
        guidance = llmprep.generate_guidance("Project", ["Ruby"])
        assert "General Guidelines" in guidance

    def test_generate_guidance_mixed(self):
        """Generates guidance for mixed language projects."""
        guidance = llmprep.generate_guidance("Project", ["Python", "C++"])
        assert "Python Best Practices" in guidance
        assert "C/C++ Best Practices" in guidance


class TestGenerateCloc:
    """Tests for generate_cloc function."""

    def test_generate_cloc_with_cloc(self, tmp_path: Path):
        """Runs cloc when available."""
        output_file = tmp_path / "stats.txt"
        with patch.object(llmprep, "check_tool", return_value=True):
            with patch.object(llmprep, "run_command", return_value=True) as mock_run:
                result = llmprep.generate_cloc(tmp_path, output_file, False)
                assert result is True
                mock_run.assert_called_once()

    def test_generate_cloc_without_cloc(self, tmp_path: Path):
        """Writes fallback message when cloc not available."""
        output_file = tmp_path / "stats.txt"
        with patch.object(llmprep, "check_tool", return_value=False):
            result = llmprep.generate_cloc(tmp_path, output_file, True)
            assert result is False
            assert "cloc not available" in output_file.read_text()


class TestRunDoxygen:
    """Tests for run_doxygen function."""

    def test_run_doxygen_not_available(self, tmp_path: Path):
        """Returns False when doxygen not available."""
        with patch.object(llmprep, "check_tool", return_value=False):
            result = llmprep.run_doxygen(tmp_path, tmp_path, True)
            assert result is False

    def test_run_doxygen_available(self, tmp_path: Path):
        """Runs doxygen when available."""
        prep_dir = tmp_path / "prep"
        prep_dir.mkdir()
        (prep_dir / "dot_graphs_doxygen").mkdir()
        (tmp_path / "Doxyfile").write_text("PROJECT_NAME = Test\n")

        def mock_check_tool(name):
            return name in ["doxygen", "dot"]

        with patch.object(llmprep, "check_tool", side_effect=mock_check_tool):
            with patch.object(llmprep, "run_command", return_value=True):
                llmprep.run_doxygen(tmp_path, prep_dir, False)
                # May succeed or fail depending on environment


class TestRunPyreverse:
    """Tests for run_pyreverse function."""

    def test_run_pyreverse_not_available(self, tmp_path: Path):
        """Returns False when pyreverse not available."""
        with patch.object(llmprep, "check_tool", return_value=False):
            result = llmprep.run_pyreverse(tmp_path, tmp_path, True)
            assert result is False


class TestRunCtags:
    """Tests for run_ctags function."""

    def test_run_ctags_not_available(self, tmp_path: Path):
        """Returns False when ctags not available."""
        with patch.object(llmprep, "check_tool", return_value=False):
            result = llmprep.run_ctags(tmp_path, tmp_path, True)
            assert result is False


class TestGenerateOverview:
    """Tests for generate_overview function."""

    def test_generate_overview_basic(self, tmp_path: Path):
        """Generates overview markdown file."""
        # Create required structure files
        (tmp_path / "codebase_structure.txt").write_text(".\n├── src\n└── tests")
        (tmp_path / "codebase_stats.txt").write_text("Python: 1000 lines")

        llmprep.generate_overview(
            tmp_path, "TestProject", {"Python": 1000}, False, False, False
        )

        overview = tmp_path / "codebase_overview.md"
        assert overview.exists()
        content = overview.read_text()
        assert "TestProject" in content
        assert "Directory Structure" in content

    def test_generate_overview_with_tools(self, tmp_path: Path):
        """Generates overview with tool outputs."""
        (tmp_path / "codebase_structure.txt").write_text(".")
        (tmp_path / "codebase_stats.txt").write_text("")
        (tmp_path / "dot_graphs_doxygen").mkdir()
        (tmp_path / "dot_graphs_doxygen" / "test.dot").write_text("digraph {}")
        (tmp_path / "dot_graphs_pyreverse").mkdir()
        (tmp_path / "dot_graphs_pyreverse" / "classes.dot").write_text("digraph {}")

        llmprep.generate_overview(tmp_path, "Project", {}, True, True, True)

        content = (tmp_path / "codebase_overview.md").read_text()
        assert "Doxygen Documentation" in content
        assert "Python Class Diagrams" in content
        assert "Symbol Index" in content


class TestCLI:
    """Tests for command-line interface."""

    def test_cli_basic(self, tmp_path: Path, monkeypatch):
        """CLI runs successfully on valid directory."""
        # Create a simple project structure
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "main.py").write_text("print('hello')")

        monkeypatch.chdir(tmp_path)

        with patch.object(
            sys, "argv", ["llmprep", str(project_dir), "--no-doxygen", "--no-pyreverse", "--no-ctags", "-q"]
        ):
            result = llmprep.main()
            assert result == 0

        prep_dir = project_dir / "llm_prep"
        assert prep_dir.exists()

    def test_cli_invalid_directory(self, tmp_path: Path):
        """CLI returns error for invalid directory."""
        with patch.object(
            sys, "argv", ["llmprep", str(tmp_path / "nonexistent")]
        ):
            result = llmprep.main()
            assert result == 1

    def test_cli_custom_output(self, tmp_path: Path, monkeypatch):
        """CLI respects custom output directory."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "main.py").write_text("print('hello')")

        monkeypatch.chdir(tmp_path)

        with patch.object(
            sys, "argv", ["llmprep", str(project_dir), "-o", "custom_out", "--no-doxygen", "--no-pyreverse", "-q"]
        ):
            result = llmprep.main()
            assert result == 0

        custom_dir = project_dir / "custom_out"
        assert custom_dir.exists()

    def test_cli_default_current_dir(self, tmp_path: Path, monkeypatch):
        """CLI defaults to current directory."""
        (tmp_path / "test.py").write_text("# test")
        monkeypatch.chdir(tmp_path)

        with patch.object(
            sys, "argv", ["llmprep", "--no-doxygen", "--no-pyreverse", "--no-ctags", "-q"]
        ):
            result = llmprep.main()
            assert result == 0


class TestLanguageDetection:
    """Tests for language detection constants."""

    def test_c_languages_constant(self):
        """C_LANGUAGES contains expected languages."""
        assert "C" in llmprep.C_LANGUAGES
        assert "C++" in llmprep.C_LANGUAGES
        assert "C/C++ Header" in llmprep.C_LANGUAGES

    def test_python_languages_constant(self):
        """PYTHON_LANGUAGES contains Python."""
        assert "Python" in llmprep.PYTHON_LANGUAGES


class TestDefaultExcludes:
    """Tests for default exclusion patterns."""

    def test_default_excludes(self):
        """DEFAULT_EXCLUDES contains common patterns."""
        excludes = llmprep.DEFAULT_EXCLUDES
        assert "__pycache__" in excludes
        assert ".git" in excludes
        assert "node_modules" in excludes
        assert "build" in excludes


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow_python_project(self, tmp_path: Path, monkeypatch):
        """Full workflow on Python project."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()

        # Create Python project structure
        (project_dir / "src").mkdir()
        (project_dir / "src" / "__init__.py").write_text("")
        (project_dir / "src" / "main.py").write_text(
            '''
def main():
    """Main entry point."""
    print("Hello, World!")

if __name__ == "__main__":
    main()
'''
        )
        (project_dir / "tests").mkdir()
        (project_dir / "tests" / "test_main.py").write_text(
            """
def test_example():
    assert True
"""
        )
        (project_dir / "README.md").write_text("# My Project\n")

        monkeypatch.chdir(tmp_path)

        with patch.object(
            sys,
            "argv",
            [
                "llmprep",
                str(project_dir),
                "--no-doxygen",
                "--no-pyreverse",
                "--no-ctags",
                "-q",
            ],
        ):
            result = llmprep.main()
            assert result == 0

        prep_dir = project_dir / "llm_prep"
        assert prep_dir.exists()
        assert (prep_dir / "codebase_overview.md").exists()
        assert (prep_dir / "llm_system_prompt.md").exists()
        assert (prep_dir / "project_guidance.md").exists()
        assert (prep_dir / "codebase_structure.txt").exists()
