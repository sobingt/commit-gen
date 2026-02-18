class GitCommitGen < Formula
  desc "AI-powered git commit message generator (local, no API key needed)"
  homepage "https://github.com/sobingt/commit-gen"
  url "https://github.com/sobingt/commit-gen/archive/refs/tags/v1.2.0.tar.gz"
  sha256 "REPLACE_WITH_SHA256_OF_YOUR_RELEASE_TARBALL"
  license "MIT"

  depends_on "python@3.11"

  def install
    # Install the run script as the main executable
    bin.install "run.sh" => "git-commit-gen"

    # Install Python deps into a virtualenv so they don't pollute the system
    venv = virtualenv_create(libexec, "python3.11")
    venv.pip_install "llama-cpp-python"
    venv.pip_install "huggingface_hub"

    # Patch the shebang to use the venv Python
    inreplace bin/"git-commit-gen", "#!/usr/bin/env python3",
              "#!#{libexec}/bin/python3"
  end

  def caveats
    <<~EOS
      After installing, download the model once (~668 MB):

        curl -fsSL https://raw.githubusercontent.com/sobingt/commit-gen/main/install.sh | bash

      Or manually run the installer to skip the brew install step and just get the model.

      Usage:
        git-commit-gen              # interactive, auto-detects changes
        git-commit-gen --install-hook   # add hook to current repo
        git commit                  # auto-generates message (after hook install)
    EOS
  end

  test do
    assert_match "git-commit-gen v", shell_output("#{bin}/git-commit-gen --version")
  end
end