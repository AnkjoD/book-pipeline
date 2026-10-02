-- Edition dimension. Grain: one row per edition_id.

with editions as (
    select *
    from {{ source('silver', 'silver_editions') }}
),

valid_books as (
    select book_id from {{ ref('dim_books') }}
),

publishers as (
    select id, name as publisher_name
    from {{ source('silver', 'silver_publishers') }}
),

languages as (
    select
        id,
        language as language_name,
        code2 as language_code
    from {{ source('silver', 'silver_languages') }}
),

countries as (
    select
        id,
        name as country_name,
        code2 as country_code
    from {{ source('silver', 'silver_countries') }}
),

reading_formats as (
    select id, format as reading_format
    from {{ source('silver', 'silver_reading_formats') }}
)

select
    e.id as edition_id,
    e.book_id,
    e.title,
    e.isbn_10,
    e.isbn_13,
    e.asin,
    e.reading_format_id,
    rf.reading_format,
    e.pages,
    case
        when e.pages between 1 and 10000 then e.pages
        else null
    end as pages_clean,
    case
        when e.pages is not null and (e.pages <= 0 or e.pages > 10000) then true
        else false
    end as is_pages_suspicious,
    e.release_year,
    case
        when e.release_year is not null
         and (
             try_cast(e.release_year as integer) < -5000
             or try_cast(e.release_year as integer) > year(current_date) + 1
         ) then true
        else false
    end as is_release_year_suspicious,
    e.language_id,
    l.language_name,
    l.language_code,
    e.country_id,
    c.country_name,
    c.country_code,
    e.publisher_id,
    p.publisher_name,
    e.rating,
    e.users_count
from editions e
inner join valid_books b
    on b.book_id = e.book_id
left join publishers p
    on p.id = e.publisher_id
left join languages l
    on l.id = e.language_id
left join countries c
    on c.id = e.country_id
left join reading_formats rf
    on rf.id = e.reading_format_id
