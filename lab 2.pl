% Profile Settings
:- set_prolog_flag(occurs_check, error).
:- set_prolog_stack(global, limit(8000000)).
:- set_prolog_stack(local,  limit(2000000)).

% PART 1
competitor(sumsum, appy).
developed(sumsum, galactica_s3).
smart_phone_tech(galactica_s3).
stole(stevey, galactica_s3, sumsum).
boss(stevey, appy).

rival(X, Y) :- competitor(X, Y).
business(T) :- smart_phone_tech(T).

unethical(X) :-
    boss(X, Y),
    stole(X, T, Z),
    business(T),
    rival(Z, Y).

% ?- trace, unethical(stevey)

% PART 2
male(charles).
male(edward).
male(andrew).
female(ann).

birth_order(charles, 1).
birth_order(edward, 4).
birth_order(ann, 2).
birth_order(andrew, 3).

get_succession_old(List) :-
    setof((G, O, N),
        ( 
        % For males: (n -> (a, birth_order, name))
        (male(N), G=a, birth_order(N, O)) ;
        % For females: (n -> (a, birth_order, name))
        (female(N), G=b, birth_order(N, O))
        ),
        List).

get_succession_new(List) :-
    setof((O, N),
        birth_order(N, O),
        List).

% ?- trace, get_succession_old(L).
% ?- trace, get_succession_new(L).
