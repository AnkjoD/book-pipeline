{{config(materialized = 'table')}}

with source_books as (

    select
        book_id,
        title,
        subtitle,
        author_names,
        author_credits,
        series_info,

        genres,
        general_tags,
        moods,
        pace,
        content_warnings,

        literary_type,
        book_category,

        release_year,
        release_year_clean,

        pages,
        pages_clean,

        rating,
        ratings_count,

        description,
        slug,

        is_title_missing,
        is_pages_suspicious,
        is_release_year_suspicious,

        -- Hiện tại chunk_text thực chất là toàn bộ document.
        nullif(trim(chunk_text), '') as document_text,

        -- Không tính lại quality ở đây.
        content_quality as document_quality,

        updated_at

    from {{ ref('dim_books') }}

),

hashed_documents as (

    select
        *,

        length(document_text) as document_length,

        -- Chuẩn hóa khoảng trắng trước khi tạo hash.
        md5(
            regexp_replace(
                lower(trim(document_text)),
                '[[:space:]]+',
                ' ',
                'g'
            )
        ) as document_hash

    from source_books

    where document_text is not null

),

ranked_documents as (

    select
        *,

        row_number() over (
            partition by document_hash
            order by
                ratings_count desc nulls last,
                book_id
        ) as document_duplicate_rank

    from hashed_documents

)

select
    *,

    case
        when document_quality = 'Rich'
         and is_title_missing = false
         and document_length >= 200
         and document_duplicate_rank = 1
        then true
        else false
    end as is_embedding_candidate

from ranked_documents