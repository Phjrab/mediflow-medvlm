# Data governance

## Data classification

Clinical images, DICOM headers, free-text clinical notes, direct identifiers,
and identifiers that can be linked back to a person are sensitive data. They are
not test fixtures and must not be committed to this repository.

## Required controls

- Perform institution-approved de-identification before creating an
  `InferenceRequest`.
- Use a pseudonymous `study_id`; retain the re-identification mapping only in an
  approved internal system.
- Pass only allowlisted metadata needed for the task. Do not pass patient name,
  MRN, birth date, accession number, original DICOM UID, or free-text notes.
- Keep image storage separate from application logs. Encrypt data at rest and in
  transit and apply least-privilege access controls.
- Define retention, deletion, backup, incident response, and third-party data
  transfer policies with the owning institution before deployment.

## Logging baseline

Permitted operational fields are a request ID, pseudonymous study ID, task,
model and prompt versions, duration, status, and non-sensitive error code. Raw
pixels, URLs with access tokens, questions, clinical reports, and model output
are excluded from standard logs.

## Limit of code checks

The request validator catches only a small set of obvious identifiers. It does
not de-identify DICOM or guarantee absence of PHI. Data protection responsibility
remains with the ingest system and the deploying organization.
