-- Many-to-many relationship between books and authors.
-- Grain: one row per (book_id, author_id).

with relationships as (
    select
        book_id,
        author_id,
        nullif(trim(role), '') as role
    from {{ source('silver', 'silver_book_authors') }}
    where book_id is not null
      and author_id is not null
),

valid_books as (
    select book_id
    from {{ ref('dim_books') }}
),

valid_authors as (
    select author_id
    from {{ ref('dim_authors') }}
)

select
    r.book_id,
    r.author_id,
    string_agg(
        distinct r.role,
        ', ' order by r.role
    ) as roles
from relationships r
inner join valid_books b
    on r.book_id = b.book_id
inner join valid_authors a
    on r.author_id = a.author_id
group by r.book_id, r.author_id
