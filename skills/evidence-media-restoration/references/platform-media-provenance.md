# Provenance of platform-hosted video

## Purpose

Determine what an online publication can and cannot establish about a video's
capture, editing, commissioning, upload, platform processing, and source-file
custody. A public post usually exposes a delivery derivative, not the camera
original or editing project.

## Identity separation

Never collapse these roles without an explicit evidence edge:

- `capture_operator`: person or crew operating the camera;
- `production_company`: external studio, agency, or internal media unit;
- `editor`: person or system assembling the published cut;
- `commissioning_unit`: corporate communications, site management, event team,
  or another client;
- `post_author`: account that published or reposted the asset;
- `platform_processor`: service that transcoded, captioned, packaged, or served
  the public derivative;
- `custodian`: organization or person holding the master, rushes, or project.

An employee publishing a video is not evidence that the employee filmed or
edited it. A company logo is not a production credit.

## Evidence ladder

Record every observation with its exact source and classify it:

1. `OBSERVED`: visible credit slate, spoken attribution, public post text,
   page metadata, caption file, container/stream tag, official event programme,
   or a creator's dated portfolio entry.
2. `DERIVED_DETERMINISTIC`: decoded timestamp candidate, hash, stream inventory,
   frame extraction, or cross-video fingerprint.
3. `RECONSTRUCTED_MINIMUM`: role attribution supported by explicit, dated,
   mutually compatible artifacts.
4. `MODEL_SUGGESTION`: vendor or creator candidate based on style, equipment,
   social graph, or similarity without an explicit credit.
5. `UNRESOLVED`: filming date, camera, operator, editor, master location, or
   other field not carried by the public artifact.

## Collection sequence

1. Preserve the exact downloaded derivative and, when lawful and public, the
   platform manifest, captions, thumbnail, post URL, post text, and retrieval
   time. Hash local files.
2. Inventory container, every stream, time bases, dimensions, frame rates,
   pixel/sample formats, handler names, encoder tags, creation tags, GPS, author,
   copyright, and original filename fields. Preserve raw tags before remuxing.
3. Separate metadata introduced by local normalization from source tags. A
   locally remuxed file cannot establish the platform's original container tags.
4. Inspect the opening and closing slates, lower thirds, spoken introductions,
   accessible caption files, descriptions, comments, and linked event pages for
   explicit credits.
5. Run a cross-publication census before assuming the platform copy is unique.
   Search the exact post wording, distinctive transcript sentences, event title,
   duration, thumbnail/keyframe crops, speakers, and public asset identifiers
   across the company's official site, newsroom, YouTube, Vimeo, Facebook,
   Instagram, other professional networks, event partners, and credited creator
   portfolios. Record whether each hit is the same cut, a longer cut, a teaser,
   a repost, or independent footage. Prefer the earliest dated public witness
   and preserve descriptions and credits that differ between platforms.
6. Treat 10- or 13-digit URL/path numbers as timestamp candidates only after
   deterministic conversion. Unless platform documentation proves capture
   semantics, label them `platform asset/processing time candidate`, never
   `filming date`.
7. Group all bitrate/codec variants of one upload into one witness cluster.
   Different transcodes can improve legibility but are not independent evidence
   for authorship or events.
8. Compare other official videos by explicit credit strings, named speakers,
   event/date, source filename, portfolio embed, thumbnail lineage, and stable
   production identifiers. Visual style, music, grading, typography, or camera
   movement may rank candidates but cannot confirm a vendor.
9. For a recurring-production hypothesis, require at least one explicit vendor
   or internal-team credit and a dated bridge to the target event. Repetition of
   the same credited team across separate official videos strengthens the bridge;
   reposts of one asset do not.

## Interpretation traps

- `Lavf`, `Lavc`, `libx264`, `libsvtav1`, `MainConcept`, handler names, and CDN
  packaging tags normally identify an encoding or delivery pipeline, not the
  camera operator, editor, or production company.
- `creation_time` may represent recording, export, remux, upload, or transcode.
  Report the tag literally until its semantics are corroborated.
- A platform account name establishes publication by that account, not creation.
- Automatic captions can reveal spoken names and locations but are not a
  production credit and must be checked against audio.
- Absence of embedded metadata does not mean no master or rushes exist.
- Likely corporate DAM, SharePoint, cloud, agency archive, or local storage are
  hypotheses unless a public record names the custodian and system.

## Minimum report

Provide a table with: field sought, exact observation, evidence class, artifact
location, interpretation limit, and status (`confirmed`, `candidate`, or
`unresolved`). End with the next discriminating artifact: original export,
credit slate, production invoice, event programme, agency portfolio entry, or
an authoritative statement from the commissioning organization.
