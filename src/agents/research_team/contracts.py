"""Domain contracts shared by the research-team nodes."""

from typing import Literal

from pydantic import BaseModel, Field


class ResearchTask(BaseModel):
    id: str = Field(description="Stable identifier, e.g. T1")
    question: str = Field(description="A single answerable research question")
    assignee: Literal["researcher", "analyst", "writer"] = "researcher"
    rationale: str = ""
    subject_ids: list[str] = Field(
        default_factory=list, description="该任务涉及的研究对象 ID，例如 S1、S2"
    )


class ResearchSubject(BaseModel):
    id: str = Field(description="Stable identifier, e.g. S1")
    name: str = Field(description="报告中使用的研究对象名称")
    aliases: list[str] = Field(default_factory=list, description="研究对象的别名")


class EvidenceCoverage(BaseModel):
    task_id: str = Field(description="对应 ResearchTask 的稳定 ID，例如 T1")
    question: str = Field(description="对应 ResearchTask 的原始问题")
    source_indices: list[int] = Field(
        description="支持该子问题的来源编号，使用分析输入中的 1-based 编号"
    )
    status: Literal["sufficient", "partial", "missing", "conflicting"] = Field(
        description="该子问题的证据覆盖状态"
    )
    evidence_strength: Literal["strong", "moderate", "weak", "insufficient"] = Field(
        description="现有证据支持该子问题结论的强度"
    )
    gaps: list[str] = Field(description="该子问题仍缺失的信息，没有则为空")
    reason: str = Field(description="判断覆盖状态的简短理由")


class AspectCoverage(BaseModel):
    name: str = Field(description="对象需要回答的证据维度，例如产能、投产时间")
    status: Literal["sufficient", "partial", "missing", "conflicting"] = Field(
        description="该证据维度的覆盖状态"
    )
    evidence_strength: Literal["strong", "moderate", "weak", "insufficient"] = Field(
        description="该证据维度的证据强度"
    )
    source_indices: list[int] = Field(description="支持该维度的来源编号，使用 1-based 编号")
    gaps: list[str] = Field(description="该维度的信息缺口，没有则为空")
    reason: str = Field(description="该维度判断的简短理由")


class SubjectEvidenceCoverage(BaseModel):
    subject_id: str = Field(description="对应 ResearchSubject 的稳定 ID")
    subject: str = Field(description="研究对象名称")
    aspects: list[AspectCoverage] = Field(description="该对象各证据维度的覆盖情况")
    status: Literal["sufficient", "partial", "missing", "conflicting"] = Field(
        description="该对象的总体证据覆盖状态"
    )
    evidence_strength: Literal["strong", "moderate", "weak", "insufficient"] = Field(
        description="该对象总体证据强度"
    )
    source_indices: list[int] = Field(description="支持该对象的来源编号，使用 1-based 编号")
    gaps: list[str] = Field(description="该对象仍缺失的信息，没有则为空")
    reason: str = Field(description="该对象总体判断的简短理由")


class EvidenceStatement(BaseModel):
    statement: str = Field(description="带来源锚点的证据陈述")
    kind: Literal["point", "change", "trend", "forecast", "status"] = Field(
        description="证据陈述类型：离散点、期间变化、趋势、预测规划或状态"
    )
    period: str | None = Field(default=None, description="单个时间点或时间范围")
    periods: list[str] = Field(default_factory=list, description="该陈述涉及的时间点")
    value_type: Literal["actual", "planned", "forecast", "reported", "unknown"] = Field(
        default="unknown", description="实际、规划、预测、已报道或未知"
    )
    metric: str | None = Field(default=None, description="指标名称")
    unit: str | None = Field(default=None, description="指标单位")
    source_indices: list[int] = Field(description="支持该陈述的来源编号，使用 1-based 编号")
    trend_origin: Literal["computed", "explicit_source_statement", "unknown"] = Field(
        default="unknown", description="趋势来自数据点计算还是来源明确表述"
    )


class Source(BaseModel):
    url: str
    title: str = ""
    snippet: str = Field(default="", description="100-200 字短摘要，用于界面展示")
    summary: str = Field(default="", description="600-800 字长摘要，供模型阅读")
    content: str = Field(default="", description="Fetch 抓取的正文 Markdown")
    site_name: str = ""
    published_at: str = ""
    authority: int = Field(default=-1, ge=-1, le=4, description="-1 未知，3 以上为高权威")
    rerank_score: float = Field(default=0.0, description="0-3，1.5 以上为高质量")
    query: str = Field(default="", description="The query that surfaced this source")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="0-1, derived from authority and rerank score"
    )


__all__ = [
    "AspectCoverage",
    "EvidenceCoverage",
    "EvidenceStatement",
    "ResearchSubject",
    "ResearchTask",
    "Source",
    "SubjectEvidenceCoverage",
]
