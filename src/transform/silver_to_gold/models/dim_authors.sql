-- Author dimension. Grain: one row per author_id.

select
    id as author_id,
    coalesce(nullif(trim(name), ''), '[Unknown author]') as name,
    bio,
    gender_id,
    born_year,
    death_year,
    location,
    is_bipoc,
    is_lgbtq,
    books_count,
    slug,
    case
        when name is null or trim(name) = '' then true
        else false
    end as is_name_missing
from {{ source('silver', 'silver_authors') }}
where id is not null
