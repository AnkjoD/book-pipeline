-- Many-to-many relationship between books and tags.
-- Grain: one row per (book_id, tag_id).

with taggings as (
    select
        book_id,
        tag_id,
        spoiler
    from {{ source('silver', 'silver_book_taggings') }}
    where book_id is not null
      and tag_id is not null
),

valid_books as (
    select book_id
    from {{ ref('dim_books') }}
),

valid_tags as (
    select tag_id
    from {{ ref('dim_tags') }}
)

select
    bt.book_id,
    bt.tag_id,
    bool_or(coalesce(bt.spoiler, false)) as spoiler
from taggings bt
inner join valid_books b
    on bt.book_id = b.book_id
inner join valid_tags t
    on bt.tag_id = t.tag_id
group by bt.book_id, bt.tag_id
