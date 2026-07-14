from __future__ import annotations

import json
from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(
        0,
        str(project_root),
    )

from src.final_submission_audit import (
    run_final_submission_audit,
    write_audit_reports,
)


def main() -> int:
    audit_result = (
        run_final_submission_audit(
            project_root
        )
    )

    json_path, markdown_path = (
        write_audit_reports(
            project_root,
            audit_result,
        )
    )

    print(
        json.dumps(
            {
                "status": (
                    audit_result["status"]
                ),
                "passed_checks": (
                    audit_result[
                        "passed_checks"
                    ]
                ),
                "failed_checks": (
                    audit_result[
                        "failed_checks"
                    ]
                ),
                "json_report": str(
                    json_path.relative_to(
                        project_root
                    )
                ),
                "markdown_report": str(
                    markdown_path.relative_to(
                        project_root
                    )
                ),
            },
            indent=2,
        )
    )

    return (
        0
        if audit_result["status"]
        == "PASS"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
