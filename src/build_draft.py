"""pycapcut으로 컷 편집 + 자막이 반영된 캡컷 draft를 조립한다."""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import pycapcut as cc
from pycapcut import trange

from .subtitles import Cue
from .utils import MediaInfoResult, PipelineError, project_root

Span = Tuple[float, float]


def _s(seconds: float) -> str:
    """pycapcut의 trange/tim은 float를 초 단위로 자동 환산하지 않으므로 항상 '1.23s' 문자열로 넘긴다."""
    return f"{seconds}s"


def _declick_fade(duration: float, max_fade: float = 0.02) -> float:
    """하드컷 조각을 그냥 이어붙이면 이어지는 지점마다 "딸깍"거리는 클릭
    노이즈가 생긴다. 사람이 못 느낄 정도로 아주 짧은 페이드(기본 0.02초)만
    넣어도 없어진다. 조각이 그보다 짧으면 절반까지만 (겹치지 않게)."""
    if duration <= 0.002:
        return 0.0
    return min(max_fade, duration / 2 - 0.001)


def build_draft(
    media_path: str,
    keep_spans: List[Span],
    cues: List[Cue],
    media: MediaInfoResult,
    draft_name: str,
    capcut_cfg: dict,
    subtitle_cfg: dict,
    cut_video: bool = True,
    hq_audio_path: Optional[str] = None,
) -> str:
    """cut_video=False면 영상은 원본 그대로(무음소거) 한 클립으로 두고,
    별도 오디오 트랙(hq_audio_path)만 keep_spans대로 잘라 넣는다.
    이동 촬영본(드론 등)은 영상을 잘라내면 화면이 툭툭 끊겨 보여서, 오디오만
    타이트하게 편집하고 화면은 매끄럽게 유지하고 싶을 때 쓴다."""
    draft_folder = cc.DraftFolder(capcut_cfg["draft_folder"])
    fps = max(1, round(media.fps))

    if media.is_audio_only:
        width = capcut_cfg.get("audio_only_canvas_width", 1920)
        height = capcut_cfg.get("audio_only_canvas_height", 1080)
    else:
        width, height = media.width, media.height

    try:
        script = draft_folder.create_draft(draft_name, width, height, fps=fps, allow_replace=True)
    except PermissionError:
        raise PipelineError(
            f"'{draft_name}' 프로젝트가 CapCut에서 열려있어 덮어쓸 수 없습니다. "
            "CapCut에서 해당 프로젝트를 닫은 뒤 다시 실행해주세요."
        )

    if media.is_audio_only:
        script.add_track(cc.TrackType.audio)
        audio_material = cc.AudioMaterial(media_path)
        cumulative = 0.0
        for start, end in keep_spans:
            duration = end - start
            if duration <= 0:
                continue
            fade = _declick_fade(duration)
            segment = cc.AudioSegment(
                audio_material,
                target_timerange=trange(_s(cumulative), _s(duration)),
                source_timerange=trange(_s(start), _s(duration)),
            )
            if fade > 0:
                segment.add_fade(_s(fade), _s(fade))
            script.add_segment(segment)
            cumulative += duration
    else:
        # 오디오는 항상 별도 트랙(hq_audio_path)에 각 조각마다 짧은 페이드를 넣어
        # 붙인다 - VideoSegment에는 add_fade가 없어서(pycapcut 제약), 영상에 오디오를
        # 그대로 얹으면 컷 지점마다 클릭 노이즈가 생긴다. cut_video 여부는 "화면을
        # 자를지"만 결정하고, 오디오 처리 방식 자체는 두 모드가 항상 동일하다.
        if not hq_audio_path:
            raise PipelineError("hq_audio_path가 필요합니다.")

        script.add_track(cc.TrackType.video)
        video_material = cc.VideoMaterial(media_path)
        if cut_video:
            cumulative = 0.0
            for start, end in keep_spans:
                duration = end - start
                if duration <= 0:
                    continue
                segment = cc.VideoSegment(
                    video_material,
                    target_timerange=trange(_s(cumulative), _s(duration)),
                    source_timerange=trange(_s(start), _s(duration)),
                    volume=0.0,  # 원본 임베디드 오디오는 끄고, 별도 오디오 트랙만 들리게
                )
                script.add_segment(segment)
                cumulative += duration
        else:
            video_segment = cc.VideoSegment(
                video_material,
                target_timerange=trange(_s(0), _s(media.duration)),
                source_timerange=trange(_s(0), _s(media.duration)),
                volume=0.0,
            )
            script.add_segment(video_segment)

        script.add_track(cc.TrackType.audio)
        audio_material = cc.AudioMaterial(hq_audio_path)
        cumulative = 0.0
        for start, end in keep_spans:
            duration = end - start
            if duration <= 0:
                continue
            fade = _declick_fade(duration)
            segment = cc.AudioSegment(
                audio_material,
                target_timerange=trange(_s(cumulative), _s(duration)),
                source_timerange=trange(_s(start), _s(duration)),
            )
            if fade > 0:
                segment.add_fade(_s(fade), _s(fade))
            script.add_segment(segment)
            cumulative += duration

    script.add_track(cc.TrackType.text, track_name="자막")
    text_style = cc.TextStyle(size=subtitle_cfg.get("font_size", 8.0), color=(1.0, 1.0, 1.0), align=1)
    border = cc.TextBorder(color=(0.0, 0.0, 0.0), width=subtitle_cfg.get("border_width", 80.0))
    clip_settings = cc.ClipSettings(transform_y=-0.8)

    for start, end, text in cues:
        duration = end - start
        if duration <= 0:
            continue
        segment = cc.TextSegment(
            text,
            trange(_s(start), _s(duration)),
            style=text_style,
            border=border,
            clip_settings=clip_settings,
        )
        script.add_segment(segment, track_name="자막")

    script.save()
    return draft_name


def build_shorts_draft(
    media_path: str,
    clip_spans: List[Span],
    cues: List[Cue],
    media: MediaInfoResult,
    draft_name: str,
    capcut_cfg: dict,
    subtitle_cfg: dict,
    shorts_cfg: dict,
    title: Optional[str] = None,
) -> str:
    """롱폼 영상에서 고른 하이라이트 구간(clip_spans)들을 이어붙여, 세로 캔버스를
    꽉 채우도록(cover) 확대/중앙 크롭한 숏폼 캡컷 draft를 만든다."""
    draft_folder = cc.DraftFolder(capcut_cfg["draft_folder"])
    fps = max(1, round(media.fps))
    canvas_width = shorts_cfg.get("canvas_width", 1080)
    canvas_height = shorts_cfg.get("canvas_height", 1920)

    try:
        script = draft_folder.create_draft(draft_name, canvas_width, canvas_height, fps=fps, allow_replace=True)
    except PermissionError:
        raise PipelineError(
            f"'{draft_name}' 프로젝트가 CapCut에서 열려있어 덮어쓸 수 없습니다. "
            "CapCut에서 해당 프로젝트를 닫은 뒤 다시 실행해주세요."
        )

    branding = shorts_cfg.get("branding") or {}
    branding_on = bool(branding.get("enabled")) and bool(branding.get("logo_path"))
    footer_ratio = float(branding.get("footer_ratio", 0.0)) if branding_on else 0.0

    def _px_to_rel_y(y_px: float) -> float:
        """캔버스 상단 기준 픽셀 y좌표를 pycapcut의 transform_y 단위(중앙=0, 위=+1, 아래=-1)로 바꾼다."""
        return 1.0 - 2.0 * y_px / canvas_height

    def _capcut_scale(target_px_ratio: float, material_width: int, material_height: int) -> float:
        """CapCut은 clip scale=1.0을 소재의 실제 픽셀 크기가 아니라, 캔버스 안에 통째로
        들어오도록 자동 축소한 크기(가로세로 중 더 작게 맞춰야 하는 쪽) 기준으로 해석한다
        (실측 확인됨: scale 보정 없이 그대로 썼더니 세로 숏폼에서 가로 영상이 캔버스를
        못 채우고 상하에 검은 여백이 생겼음). 그래서 "실제 픽셀 기준으로 원하는 배율"을
        그 자동 축소 배율로 나눠서, CapCut이 다시 곱했을 때 원하는 배율이 나오게 만든다."""
        auto_fit = min(canvas_width / material_width, canvas_height / material_height)
        return target_px_ratio / auto_fit

    # 원본(보통 가로) 영상을 세로 캔버스에 빈틈없이 채우기 위해, 캔버스 가로/세로
    # 비율 중 더 많이 확대해야 하는 쪽에 맞춰 등비 확대(cover)한다. 하단에 브랜딩
    # 띠를 남겨야 하면(footer_ratio) 그만큼 뺀 위쪽 영역에만 맞춰 채우고, 그 영역의
    # 중앙으로 오도록 위로 밀어올린다.
    video_area_height = canvas_height * (1.0 - footer_ratio)
    target_px_scale = max(canvas_width / media.width, video_area_height / media.height)
    scale = _capcut_scale(target_px_scale, media.width, media.height)
    video_transform_y = _px_to_rel_y(video_area_height / 2.0) if branding_on else 0.0
    clip_settings = cc.ClipSettings(scale_x=scale, scale_y=scale, transform_y=video_transform_y)

    script.add_track(cc.TrackType.video)
    video_material = cc.VideoMaterial(media_path)
    cumulative = 0.0
    for start, end in clip_spans:
        duration = end - start
        if duration <= 0:
            continue
        segment = cc.VideoSegment(
            video_material,
            target_timerange=trange(_s(cumulative), _s(duration)),
            source_timerange=trange(_s(start), _s(duration)),
            clip_settings=clip_settings,
        )
        if branding_on:
            # 하단 띠 자리를 항상 검게 채워서(영상이 위쪽에만 있어도) 로고와 자연스럽게 이어지게 한다.
            segment.add_background_filling("color", color="#000000FF")
        script.add_segment(segment)
        cumulative += duration

    total_duration = cumulative

    if branding_on:
        logo_path = branding["logo_path"]
        if not os.path.isabs(logo_path):
            logo_path = os.path.join(project_root(), logo_path)

        script.add_track(cc.TrackType.video, track_name="브랜딩로고")
        logo_material = cc.VideoMaterial(logo_path)
        logo_width_ratio = branding.get("logo_width_ratio", 0.55)
        logo_target_px_scale = (canvas_width * logo_width_ratio) / logo_material.width
        logo_scale = _capcut_scale(logo_target_px_scale, logo_material.width, logo_material.height)

        footer_top_px = canvas_height * (1.0 - footer_ratio)
        footer_height_px = canvas_height * footer_ratio
        logo_center_px = footer_top_px + footer_height_px * branding.get("logo_center_frac", 0.32)
        logo_clip_settings = cc.ClipSettings(
            scale_x=logo_scale, scale_y=logo_scale, transform_y=_px_to_rel_y(logo_center_px)
        )
        logo_segment = cc.VideoSegment(
            logo_material,
            target_timerange=trange(_s(0), _s(total_duration)),
            clip_settings=logo_clip_settings,
        )
        script.add_segment(logo_segment, track_name="브랜딩로고")

        text_center_px = footer_top_px + footer_height_px * branding.get("text_center_frac", 0.78)
        text_transform_y = _px_to_rel_y(text_center_px)
        border = cc.TextBorder(color=(0.0, 0.0, 0.0), width=branding.get("text_border_width", 40.0))

        # 같은 트랙에는 겹치는 시간대의 자막을 두 개 못 넣으므로(전체 구간 내내 둘 다
        # 떠 있어야 함), CTA 문구와 전화번호를 서로 다른 트랙으로 분리한다.
        cta_text = (branding.get("cta_text") or "").strip()
        if cta_text:
            script.add_track(cc.TrackType.text, track_name="브랜딩안내문구")
            cta_style = cc.TextStyle(size=branding.get("cta_font_size", 5.0), color=(1.0, 1.0, 1.0), align=1)
            cta_clip_settings = cc.ClipSettings(transform_y=text_transform_y + 0.03)
            script.add_segment(
                cc.TextSegment(
                    cta_text, trange(_s(0), _s(total_duration)),
                    style=cta_style, border=border, clip_settings=cta_clip_settings,
                ),
                track_name="브랜딩안내문구",
            )

        phone = (branding.get("phone") or "").strip()
        if phone:
            script.add_track(cc.TrackType.text, track_name="브랜딩전화번호")
            phone_style = cc.TextStyle(
                size=branding.get("phone_font_size", 6.5), bold=True, color=(1.0, 0.84, 0.35), align=1
            )
            phone_clip_settings = cc.ClipSettings(transform_y=text_transform_y - 0.03)
            script.add_segment(
                cc.TextSegment(
                    phone, trange(_s(0), _s(total_duration)),
                    style=phone_style, border=border, clip_settings=phone_clip_settings,
                ),
                track_name="브랜딩전화번호",
            )

    if title and title.strip():
        script.add_track(cc.TrackType.text, track_name="제목")
        title_style = cc.TextStyle(
            size=shorts_cfg.get("title_font_size", 9.0),
            bold=True,
            color=(1.0, 1.0, 1.0),
            align=1,
            auto_wrapping=True,
        )
        title_border = cc.TextBorder(color=(0.0, 0.0, 0.0), width=shorts_cfg.get("title_border_width", 60.0))
        title_clip_settings = cc.ClipSettings(transform_y=shorts_cfg.get("title_transform_y", 0.72))
        title_segment = cc.TextSegment(
            title.strip(),
            trange(_s(0), _s(total_duration)),
            style=title_style,
            border=title_border,
            clip_settings=title_clip_settings,
        )
        script.add_segment(title_segment, track_name="제목")

    if cues:
        # 본영상에 이미 자막이 박혀있는 경우가 많아서(shorts.captions_enabled=False가
        # 기본) cues가 보통 비어있다 - 그럴 땐 빈 트랙조차 만들지 않는다.
        script.add_track(cc.TrackType.text, track_name="자막")
        text_style = cc.TextStyle(size=subtitle_cfg.get("font_size", 8.0), color=(1.0, 1.0, 1.0), align=1)
        border = cc.TextBorder(color=(0.0, 0.0, 0.0), width=subtitle_cfg.get("border_width", 80.0))
        # 브랜딩 띠가 있으면 자막이 그 위에 겹치지 않도록, 자막 위치를 "전체 캔버스"가
        # 아니라 "영상이 실제로 채워지는 위쪽 영역" 기준 하단부로 계산한다.
        # footer_ratio=0(브랜딩 없음)일 때는 video_area_height=canvas_height가 되어
        # 기존 transform_y=-0.8과 정확히 동일한 위치가 나온다 (caption_bottom_frac=0.9 기준).
        caption_bottom_frac = shorts_cfg.get("caption_bottom_frac", 0.9)
        caption_transform_y = _px_to_rel_y(video_area_height * caption_bottom_frac)
        text_clip_settings = cc.ClipSettings(transform_y=caption_transform_y)

    for start, end, text in cues:
        duration = end - start
        if duration <= 0:
            continue
        segment = cc.TextSegment(
            text,
            trange(_s(start), _s(duration)),
            style=text_style,
            border=border,
            clip_settings=text_clip_settings,
        )
        script.add_segment(segment, track_name="자막")

    script.save()
    return draft_name
