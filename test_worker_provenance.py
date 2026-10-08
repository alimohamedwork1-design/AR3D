import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent

class WorkerBuildProvenanceTests(unittest.TestCase):
    def test_container_image_uses_source_sha_from_gh_actions(self):
        docker = (ROOT / "Dockerfile").read_text()
        workflow = (ROOT / ".github/workflows/build.yml").read_text()
        self.assertIn("ARG ARQARY_BUILD_SHA=unknown", docker)
        self.assertIn("ENV ARQARY_WORKER_BUILD_SHA=${ARQARY_BUILD_SHA}", docker)
        self.assertIn("ARQARY_BUILD_SHA=${{ github.sha }}", workflow)
        self.assertIn("ghcr.io/alimohamedwork1-design/ar3d:${{ github.sha }}", workflow)

    def test_both_success_and_failure_return_revision_for_endpoint_parity(self):
        worker = (ROOT / "handler.py").read_text()
        self.assertGreaterEqual(worker.count('"worker_build_sha": WORKER_BUILD_SHA'), 2)
        self.assertIn('[worker] revision={WORKER_BUILD_SHA}', worker)

if __name__ == "__main__":
    unittest.main()
